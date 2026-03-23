"""
Visualization Runner
Main script to generate all visualizations and reports from simulation data.
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "src" / "analysis"))
sys.path.insert(0, str(project_root / "src" / "visualization"))

# Direct imports
from metrics import MetricsCalculator
from reporter import ResearchReporter
from heatmaps import HeatmapVisualizer
from networks import NetworkVisualizer
from timelines import TimelineVisualizer
from dashboards import DashboardVisualizer


def print_header(text: str):
    """Print a nice header"""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80)


def main():
    """
    Main visualization pipeline.
    
    Steps:
    1. Calculate all metrics
    2. Generate text reports
    3. Create heatmaps
    4. Create network graphs
    5. Create timelines
    6. Create dashboards
    """
    
    print("\n" + "🎨 "*20)
    print("SOCIAL NETWORK SIMULATION - VISUALIZATION & ANALYSIS")
    print("🎨 "*20)
    
    # ── STEP 1: CALCULATE METRICS ─────────────────────────────────────
    print_header("STEP 1: Calculating Metrics")
    
    metrics_calc = MetricsCalculator(data_dir="data")
    stats = metrics_calc.generate_summary_statistics()
    
    if 'error' in stats:
        print(f"\n❌ Error: {stats['error']}")
        print("   Make sure you've run a simulation first (python src/main.py)")
        return
    
    print("\n✅ Metrics calculated successfully!")
    
    # ── STEP 2: GENERATE REPORTS ──────────────────────────────────────
    print_header("STEP 2: Generating Reports")
    
    reporter = ResearchReporter(output_dir="visualizations/reports")
    
    # Markdown report
    md_report = reporter.generate_markdown_report(stats)
    print(f"\n📄 Markdown report: {md_report}")
    
    # JSON metrics
    json_metrics = reporter.save_json_metrics(stats)
    print(f"💾 JSON metrics: {json_metrics}")
    
    # ── STEP 3: CREATE HEATMAPS ───────────────────────────────────────
    print_header("STEP 3: Creating Similarity Heatmaps")
    
    heatmap_viz = HeatmapVisualizer(output_dir="visualizations/heatmaps")
    
    # Individual heatmaps for each round
    heatmap_paths = heatmap_viz.create_all_heatmaps(data_dir="data/similarity")
    print(f"\n✅ Created {len(heatmap_paths)} heatmaps")
    
    # Animated GIF
    gif_path = heatmap_viz.create_animated_heatmap(
        data_dir="data/similarity",
        duration=1500  # 1.5 seconds per frame
    )
    
    # Difference heatmaps (if we have coalition events)
    if stats.get('coalition_events', 0) >= 2:
        # Show difference between rounds 2 and 4 (before and after re-clustering)
        diff_path = heatmap_viz.create_difference_heatmap(
            round_before=2,
            round_after=4,
            data_dir="data/similarity"
        )
    
    # ── STEP 4: CREATE NETWORK GRAPHS ─────────────────────────────────
    print_header("STEP 4: Creating Network Graphs")
    
    network_viz = NetworkVisualizer(output_dir="visualizations/networks")
    
    # Individual networks for each round
    network_paths = network_viz.create_all_network_graphs(
        similarity_threshold=0.4  # Only show edges with similarity > 0.4
    )
    print(f"\n✅ Created {len(network_paths)} network graphs")
    
    # Comparison networks (before/after clustering)
    if stats.get('coalition_events', 0) >= 1:
        comparison_path = network_viz.create_comparison_network(
            round_before=1,
            round_after=2,  # After first clustering
            similarity_threshold=0.4
        )
        
        if stats.get('coalition_events', 0) >= 2:
            comparison_path2 = network_viz.create_comparison_network(
                round_before=2,
                round_after=4,  # After second clustering
                similarity_threshold=0.4
            )
    
    # ── STEP 5: CREATE TIMELINES ──────────────────────────────────────
    print_header("STEP 5: Creating Timeline Visualizations")
    
    timeline_viz = TimelineVisualizer(output_dir="visualizations/timelines")
    
    # Coalition membership timeline
    timeline_path = timeline_viz.create_coalition_timeline()
    
    # Migration flow
    migration_path = timeline_viz.create_migration_sankey()
    
    # Stability metrics chart
    stability_path = timeline_viz.create_stability_chart()
    
    # ── STEP 6: CREATE DASHBOARDS ─────────────────────────────────────
    print_header("STEP 6: Creating Dashboards")
    
    dashboard_viz = DashboardVisualizer(output_dir="visualizations/dashboards")
    
    # Agent activity dashboard
    activity_path = dashboard_viz.create_agent_activity_dashboard()
    
    # Similarity evolution chart
    evolution_path = dashboard_viz.create_similarity_evolution_chart(
        similarity_dir="data/similarity"
    )
    
    # Overall summary dashboard
    summary_path = dashboard_viz.create_summary_dashboard(stats)
    
    # ── FINAL SUMMARY ──────────────────────────────────────────────────
    print_header("VISUALIZATION COMPLETE!")
    
    print("\n📁 All visualizations saved to:")
    print(f"\n   📊 Heatmaps:     visualizations/heatmaps/")
    print(f"   🕸️  Networks:    visualizations/networks/")
    print(f"   📅 Timelines:    visualizations/timelines/")
    print(f"   📈 Dashboards:   visualizations/dashboards/")
    print(f"   📄 Reports:      visualizations/reports/")
    
    print("\n✨ Key Files to Review:")
    print(f"\n   1. Summary Report:     {md_report.name}")
    print(f"   2. Animated Heatmap:   similarity_evolution.gif")
    print(f"   3. Coalition Timeline: coalition_timeline.png")
    print(f"   4. Summary Dashboard:  summary_dashboard.png")
    
    print("\n🎓 Research Outputs:")
    print("\n   → Use heatmaps to show similarity evolution")
    print("   → Use network graphs to visualize coalition structure")
    print("   → Use timeline to demonstrate stability/churn")
    print("   → Use dashboard for summary statistics")
    print("   → Include markdown report in your documentation")
    
    print("\n" + "="*80)
    print("✅ All done! Your visualizations are ready for analysis.")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()