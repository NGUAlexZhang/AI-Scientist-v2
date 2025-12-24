import argparse
import atexit
import logging
import shutil
import json
import pickle
from types import SimpleNamespace

from omegaconf import OmegaConf

from . import backend
from .journal import Journal, Node
from .journal2report import journal2report
from rich.columns import Columns
from rich.console import Group
from rich.live import Live
from rich.padding import Padding
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)
from rich.text import Text
from rich.status import Status
from rich.tree import Tree
from .utils.config import load_task_desc, prep_agent_workspace, save_run, load_cfg
from .utils import serialize
from .agent_manager import AgentManager
from pathlib import Path
from .agent_manager import Stage
from .log_summarization import overall_summarize


logger = logging.getLogger("ai-scientist")


def _load_cfg_from_restore_dir(restore_dir: Path, default_cfg_path: Path):
    """Load configuration when resuming from an existing log directory."""
    stage_cfgs = sorted(restore_dir.glob("stage_*/config.yaml"))
    cfg_path = stage_cfgs[-1] if stage_cfgs else restore_dir / "config.yaml"

    if cfg_path.exists():
        cfg = OmegaConf.load(cfg_path)
    else:
        cfg = load_cfg(default_cfg_path)

    # Ensure paths point to the restored run so report outputs land in the right place
    cfg.log_dir = Path(restore_dir)
    if hasattr(cfg, "workspace_dir"):
        cfg.workspace_dir = Path(cfg.workspace_dir)
    else:
        cfg.workspace_dir = Path(restore_dir)
    return cfg


def _restore_journals_from_dir(restore_dir: Path):
    """Reconstruct journals from saved stage folders."""
    stage_dirs = [p for p in restore_dir.iterdir() if p.is_dir() and p.name.startswith("stage_")]
    stage_dirs = sorted(stage_dirs, key=lambda p: int(p.name.split("_")[1]) if len(p.name.split("_")) > 1 and p.name.split("_")[1].isdigit() else 0)

    journals = {}
    for stage_dir in stage_dirs:
        journal_path = stage_dir / "journal.json"
        if not journal_path.exists():
            continue
        try:
            journal = serialize.load_json(journal_path, Journal)
            stage_name = stage_dir.name.replace("stage_", "", 1)
            journals[stage_name] = journal
        except Exception as e:
            logger.warning(f"Failed to load journal from {journal_path}: {e}")
    return journals


def _restore_manager_from_dir(restore_dir: Path, cfg):
    """Restore manager or journals from a previously saved run directory."""
    manager_pickle = restore_dir / "manager.pkl"
    if manager_pickle.exists():
        try:
            with open(manager_pickle, "rb") as f:
                restored = pickle.load(f)
            if isinstance(restored, AgentManager):
                restored.cfg = cfg
                return restored
            if isinstance(restored, list):
                journals = dict(restored)
                return SimpleNamespace(
                    journals=journals,
                    completed_stages=list(journals.keys()),
                    current_stage=None,
                )
        except Exception as e:
            logger.warning(f"Failed to load manager.pkl from {restore_dir}: {e}")

    journals = _restore_journals_from_dir(restore_dir)
    return SimpleNamespace(
        journals=journals,
        completed_stages=list(journals.keys()),
        current_stage=None,
    )


def journal_to_rich_tree(journal: Journal, cfg):
    best_node = journal.get_best_node(cfg=cfg)

    def append_rec(node: Node, tree):
        if node.is_buggy:
            s = "[red]◍ bug"
        else:
            style = "bold " if node is best_node else ""

            if node is best_node:
                s = f"[{style}green]● {node.metric.value:.3f} (best)"
            else:
                s = f"[{style}green]● {node.metric.value:.3f}"

        subtree = tree.add(s)
        for child in node.children:
            append_rec(child, subtree)

    tree = Tree("[bold blue]Solution tree")
    for n in journal.draft_nodes:
        append_rec(n, tree)
    return tree


def perform_experiments_bfts(config_path: str, restore_dir: str | None = None):
    # turn config path string into a path object
    config_path = Path(config_path)
    restore_path = Path(restore_dir) if restore_dir else None

    if restore_path:
        cfg = _load_cfg_from_restore_dir(restore_path, config_path)
        logger.info(f'Restoring run from "{restore_path}"; skipping execution and regenerating reports if requested.')
        manager = _restore_manager_from_dir(restore_path, cfg)
    else:
        cfg = load_cfg(config_path)
        logger.info(f'Starting run "{cfg.exp_name}"')

        task_desc = load_task_desc(cfg)
        print(task_desc)
        task_desc_str = backend.compile_prompt_to_md(task_desc)

        global_step = 0

        with Status("Preparing agent workspace (copying and extracting files) ..."):
            prep_agent_workspace(cfg)

        def cleanup():
            if global_step == 0:
                shutil.rmtree(cfg.workspace_dir)

        atexit.register(cleanup)

        manager = AgentManager(
            task_desc=task_desc,
            cfg=cfg,
            workspace_dir=Path(cfg.workspace_dir),
        )

        prog = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=20),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
        )
        status = Status("[green]Running experiments...")
        prog.add_task("Progress:", total=cfg.agent.steps, completed=global_step)

        def create_exec_callback(status_obj):
            def exec_callback(*args, **kwargs):
                status_obj.update("[magenta]Executing code...")
                res = interpreter.run(*args, **kwargs)
                status_obj.update("[green]Generating code...")
                return res

            return exec_callback

        def step_callback(stage, journal):
            print("Step complete")
            try:
                # Generate and save notes for this step
                notes_dir = cfg.log_dir / f"stage_{stage.name}" / "notes"
                notes_dir.mkdir(parents=True, exist_ok=True)

                # Save latest node summary
                if journal.nodes:
                    latest_node = journal.nodes[-1]
                    if hasattr(latest_node, "_agent"):
                        summary = latest_node._agent._generate_node_summary(latest_node)
                        with open(
                            notes_dir / f"node_{latest_node.id}_summary.json", "w"
                        ) as f:
                            json.dump(summary, f, indent=2)


                if cfg.agent.get("summary", None) is not None:
                    current_findings = journal.generate_summary(
                        include_code=False, 
                        **{
                            "model": cfg.agent.summary.model, 
                            "temp": cfg.agent.summary.temp
                        }
                    )
                else:
                    current_findings = journal.generate_summary(include_code=False)

                best_metric = journal.get_best_node(cfg=cfg)

                # Generate and save stage progress summary
                stage_summary = {
                    "stage": stage.name,
                    "total_nodes": len(journal.nodes),
                    "buggy_nodes": len(journal.buggy_nodes),
                    "good_nodes": len(journal.good_nodes),
                    "best_metric": (
                        str(best_metric.metric)
                        if best_metric
                        else "None"
                    ),
                    "current_findings": current_findings,
                }

                with open(notes_dir / "stage_progress.json", "w") as f:
                    json.dump(stage_summary, f, indent=2)

                # Save the run as before
                save_run(cfg, journal, stage_name=f"stage_{stage.name}")

            except Exception as e:
                print(f"Error in step callback: {e}")

            print(f"Run saved at {cfg.log_dir / f'stage_{stage.name}'}")
            print(f"Step {len(journal)}/{stage.max_iterations} at stage_{stage.name}")
            print(f"Run saved at {cfg.log_dir / f'stage_{stage.name}'}")

        def generate_live(manager):
            current_stage = manager.current_stage
            current_journal = manager.journals.get(
                current_stage.name if current_stage else None, None
            )

            if current_journal:
                tree = journal_to_rich_tree(current_journal, cfg)
            else:
                tree = Tree("[bold blue]No results yet")

            file_paths = [
                f"Result visualization:\n[yellow]▶ {str((cfg.log_dir / 'tree_plot.html'))}",
                f"Agent workspace directory:\n[yellow]▶ {str(cfg.workspace_dir)}",
                f"Experiment log directory:\n[yellow]▶ {str(cfg.log_dir)}",
            ]

            stage_info = [
                "[bold]Experiment Progress:",
                f"Current Stage: [cyan]{current_stage.name if current_stage else 'None'}[/cyan]",
                f"Completed Stages: [green]{', '.join(manager.completed_stages)}[/green]",
            ]

            left = Group(
                Panel(Text(task_desc_str.strip()), title="Task description"),
                Panel(Text("\n".join(stage_info)), title="Stage Progress"),
                prog,
                status,
            )
            right = tree
            wide = Group(*file_paths)

            return Panel(
                Group(
                    Padding(wide, (1, 1, 1, 1)),
                    Columns(
                        [Padding(left, (1, 2, 1, 1)), Padding(right, (1, 1, 1, 2))],
                        equal=True,
                    ),
                ),
                title=f'[b]AIDE is working on experiment: [bold green]"{cfg.exp_name}[/b]"',
                subtitle="Press [b]Ctrl+C[/b] to stop the run",
            )

        live = Live(
            generate_live(manager),
            refresh_per_second=16,
            screen=True,
        )

        manager.run(exec_callback=create_exec_callback(status), step_callback=step_callback)

        manager_pickle_path = cfg.log_dir / "manager.pkl"
        try:
            with open(manager_pickle_path, "wb") as f:
                pickle.dump(manager, f)
            logger.info(f"Saved manager state to: {manager_pickle_path}")
        except Exception as e:
            logger.warning(f"Failed to save full manager state: {e}")
            try:
                with open(manager_pickle_path, "wb") as f:
                    pickle.dump(manager.journals.items(), f)
                logger.info(f"Saved manager journals to: {manager_pickle_path}")
            except Exception as e:
                logger.error(f"Failed to save manager journals: {e}")

    if getattr(cfg, "generate_report", True):
        print("Generating final report from all stages...")
        (
            draft_summary,
            baseline_summary,
            research_summary,
            ablation_summary,
        ) = overall_summarize(manager.journals.items(), cfg)
        draft_summary_path = cfg.log_dir / "draft_summary.json"
        baseline_summary_path = cfg.log_dir / "baseline_summary.json"
        research_summary_path = cfg.log_dir / "research_summary.json"
        ablation_summary_path = cfg.log_dir / "ablation_summary.json"

        with open(draft_summary_path, "w") as draft_file:
            json.dump(draft_summary, draft_file, indent=2)

        with open(baseline_summary_path, "w") as baseline_file:
            json.dump(baseline_summary, baseline_file, indent=2)

        with open(research_summary_path, "w") as research_file:
            json.dump(research_summary, research_file, indent=2)

        with open(ablation_summary_path, "w") as ablation_file:
            json.dump(ablation_summary, ablation_file, indent=2)

        print(f"Summary reports written to files:")
        print(f"- Draft summary: {draft_summary_path}")
        print(f"- Baseline summary: {baseline_summary_path}")
        print(f"- Research summary: {research_summary_path}")
        print(f"- Ablation summary: {ablation_summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run or resume AI Scientist experiments.")
    parser.add_argument(
        "--config",
        dest="config_path",
        default="treesearch/utils/config.yaml",
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--restore-dir",
        dest="restore_dir",
        default=None,
        help="Existing log directory to restore and regenerate reports from.",
    )

    args = parser.parse_args()
    perform_experiments_bfts(args.config_path, restore_dir=args.restore_dir)
