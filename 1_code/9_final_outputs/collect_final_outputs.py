"""
collect_final_outputs.py — bundle manuscript tables + figures into 4_outputs/final/.

Entry point: ``python main.py --get-outputs-final``

This is a pure read-derived convenience step. It NEVER overwrites any canonical
artifact: it only reads the existing MPNet-track tex tables and figure images and
writes a human-facing copy into ``4_outputs/final/``:

    * TableX.xlsx          — one Excel workbook per manuscript table (tex -> xlsx)
    * FigX.png / FigX.pdf  — one copy per manuscript figure, by PRINTED PDF order
    * tables_tex/<orig>.tex— the original .tex for traceability
    * README.md            — cross-reference: each TableX/FigX -> origin -> scripts

The bundle is MPNet-only (the canonical manuscript track). Figures are named by
their printed order in dissertation.pdf (not by their source filename), so the
files match what a reader sees.

Cross-references below ("generator" = script that writes the output file;
"data" = script that produces the underlying data when different) are mirrored
into 4_outputs/final/README.md at runtime.

# ----- MANIFEST (tables) ---------------------------------------------------
# Table18   tab18_corpus_provenance.tex
#           generator: 1_code/7_main_analysis/2_appendix/export_corpus_provenance.py
#           data:      1_code/2_segment (hydrated snapshot counts)
# Table16   tab16_model_selection_ranking.tex
#           generator: 1_code/7_main_analysis/2_appendix/d1_export_model_selection_nums.py
#           data:      1_code/4_supervised_model_train/1_grid_search.py
# Table1    tab1_classifier_performance.tex
#           generator: 1_code/7_main_analysis/1_main_text/3_generate_cross_sensitivity_table.py
#           data:      1_code/5_supervised_model_infer/score_supervised.py
# Table5    tab5_register_decomposition.tex
#           generator: 1_code/7_main_analysis/0_shared/g_register_decomposition.py
#           data:      1_code/7_main_analysis/1_main_text/1_semantic_gap.py
# Table4    tab4_interaction_h25.tex
#           generator: 1_code/7_main_analysis/1_main_text/2_coverage_semantic_interaction.py
#           data:      1_code/5_supervised_model_infer/score_supervised.py
# Table10   tab10_concept_coverage.tex
#           generator: 1_code/7_main_analysis/1_main_text/3_generate_cross_sensitivity_table.py
#           data:      1_code/7_main_analysis/1_main_text/1_semantic_gap.py
# Table11   tab_concept_reference.tex
#           generator: 1_code/7_main_analysis/0_shared/g_register_decomposition.py
#           data:      1_code/7_main_analysis/1_main_text/1_semantic_gap.py
# TableA3   tab_a3_sdg4_lexical_audit.tex
#           generator: 1_code/7_main_analysis/2_appendix/a3_sdg4_lexical_audit.py
#           data:      1_code/5_supervised_model_infer/score_supervised.py
# TableB2   tab_b2_semantic_gap_interpret_all.tex
#           generator: 1_code/7_main_analysis/2_appendix/b2_semantic_gap_text_interpretability.py
#           data:      1_code/7_main_analysis/1_main_text/1_semantic_gap.py
# Table13 (+_cont) tab13_distributional_gap.tex  (two tabular blocks)
#           generator: 1_code/7_main_analysis/1_main_text/g_distributional_gap.py
#           data:      1_code/5_supervised_model_infer/score_supervised.py
# Table14   tab14_distributional_h1.tex
#           generator: 1_code/7_main_analysis/1_main_text/g_distributional_h1_correlation.py
#           data:      1_code/7_main_analysis/1_main_text/g_distributional_gap.py
# TableC    tab_c_sample_stability.tex
#           generator: 1_code/7_main_analysis/2_appendix/c_sample_stability.py
#           data:      1_code/5_supervised_model_infer/score_supervised.py
# Table16_full tab16_model_selection_full.tex
#           generator: 1_code/7_main_analysis/2_appendix/d1_export_model_selection_nums.py
#           data:      1_code/4_supervised_model_train/1_grid_search.py
# Table12_cross tab12_register_cross.tex
#           generator: 1_code/7_main_analysis/2_appendix/f3_register_iterative_cross_table.py
#           data:      1_code/7_main_analysis/0_shared/register_adjust.py
# TableA1_design tab_a1_register_sample_design.tex
#           generator: 1_code/7_main_analysis/2_appendix/a1_register_validation.py
#           data:      1_code/7_main_analysis/0_shared/register_adjust.py
# TableA1_features tab_a1_register_features.tex
#           generator: 1_code/7_main_analysis/2_appendix/a1_register_validation.py
#           data:      1_code/7_main_analysis/0_shared/register_adjust.py
# TableA1   tab_a1_register_validation.tex
#           generator: 1_code/7_main_analysis/2_appendix/a1_register_validation.py
#           data:      1_code/7_main_analysis/0_shared/register_adjust.py
# TableA1_selectivity tab_a1_register_validation_selectivity.tex
#           generator: 1_code/7_main_analysis/2_appendix/a1_register_validation.py
#           data:      1_code/7_main_analysis/0_shared/register_adjust.py
# Table6a   tab6a_cross_sensitivity.tex
#           generator: 1_code/7_main_analysis/1_main_text/3_generate_cross_sensitivity_table.py
#           data:      1_code/5_supervised_model_infer/score_supervised.py
# Table6b   tab6b_cross_sensitivity.tex
#           generator: 1_code/7_main_analysis/1_main_text/3_generate_cross_sensitivity_table.py
#           data:      1_code/5_supervised_model_infer/score_supervised.py
# Table7a   tab7a_encoder_sensitivity.tex
#           generator: 1_code/7_main_analysis/1_main_text/3_generate_cross_sensitivity_table.py
#           data:      1_code/5_supervised_model_infer/score_supervised.py
# Table7b   tab7b_encoder_sensitivity.tex
#           generator: 1_code/7_main_analysis/1_main_text/3_generate_cross_sensitivity_table.py
#           data:      1_code/5_supervised_model_infer/score_supervised.py
# Table9    tab9_encoder_sensitivity_coverage.tex
#           generator: 1_code/7_main_analysis/1_main_text/3_generate_cross_sensitivity_table.py
#           data:      1_code/5_supervised_model_infer/score_supervised.py
# Table8    tab8_coverage_sensitivity.tex
#           generator: 1_code/7_main_analysis/1_main_text/3_generate_cross_sensitivity_table.py
#           data:      1_code/7_main_analysis/1_main_text/2_coverage_semantic_interaction.py
# Table_minilm_ref tab_minilm_reference.tex
#           generator: 1_code/7_main_analysis/0_shared/g_register_decomposition.py
#           data:      1_code/7_main_analysis/1_main_text/1_semantic_gap.py
# Table_scibert_ref tab_scibert_reference.tex
#           generator: 1_code/7_main_analysis/0_shared/g_register_decomposition.py
#           data:      1_code/7_main_analysis/1_main_text/1_semantic_gap.py
# TableH1_covgap tab_app_cross_method_covgap.tex
#           generator: 1_code/7_main_analysis/2_appendix/h1_cross_method_gap_values.py
#           data:      1_code/7_main_analysis/1_main_text/2_coverage_semantic_interaction.py
# TableH1_semgap tab_app_cross_method_semgap.tex
#           generator: 1_code/7_main_analysis/2_appendix/h1_cross_method_gap_values.py
#           data:      1_code/7_main_analysis/1_main_text/1_semantic_gap.py
# TableI1   tab_app_assignment_method_comparison.tex
#           generator: 1_code/7_main_analysis/2_appendix/i1_assignment_method_comparison.py
#           data:      1_code/5_supervised_model_infer/score_supervised.py
# TableJ1   tab_j1_raw_value_correlation.tex
#           generator: 1_code/7_main_analysis/2_appendix/j1_raw_value_correlation.py
#           data:      1_code/7_main_analysis/1_main_text/2_coverage_semantic_interaction.py
# TableA2   tab_a2_policy_source_family_combined.tex
#           generator: 1_code/7_main_analysis/2_appendix/a2_policy_source_family_sensitivity.py
#           data:      1_code/7_main_analysis/1_main_text/2_coverage_semantic_interaction.py
# TableA2_h25 tab_a2_policy_source_family_h25.tex
#           generator: 1_code/7_main_analysis/2_appendix/a2_policy_source_family_sensitivity.py
#           data:      1_code/7_main_analysis/1_main_text/2_coverage_semantic_interaction.py
# TableK1   tab_k1_specification_grid.tex
#           generator: 1_code/7_main_analysis/2_appendix/k1_regression_semantic_gap.py
#           data:      1_code/7_main_analysis/1_main_text/1_semantic_gap.py
#
# ----- MANIFEST (figures, by printed PDF order) -----------------------------
# Fig1  fig1_conceptual_framework  generator: 1_code/8_visualization/build_conceptual_figs.py
# Fig2  fig6_pipeline_flowchart    generator: 1_code/8_visualization/build_pipeline_flowchart.py
# Fig3  fig2_coverage_profiles     generator: 1_code/8_visualization/plot_figures.py
#                                 data: 1_code/7_main_analysis/1_main_text/2_coverage_semantic_interaction.py
# Fig4  fig3_pca_register_before_after generator: 1_code/7_main_analysis/1_main_text/0_pca_register_before_after.py
#                                 data: 1_code/7_main_analysis/0_shared/register_adjust.py
# Fig5  fig4_semantic_gap          generator: 1_code/8_visualization/plot_figures.py
#                                 data: 1_code/7_main_analysis/1_main_text/1_semantic_gap.py
# Fig6a..Fig6d fig9_h1a..h1d       generator: 1_code/8_visualization/plot_figures.py
#                                 data: 1_code/7_main_analysis/1_main_text/2_coverage_semantic_interaction.py
# Fig7  fig8_centroid_similarity_heatmap generator: 1_code/8_visualization/plot_figures.py
#                                 data: 1_code/6_calculate_centroids/1_build_centroid_similarity_matrix.py
# ---------------------------------------------------------------------------
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

import pandas as pd
from openpyxl.styles import Font

REPO_ROOT = Path(__file__).resolve().parents[2]

# Each table: final handle, LaTeX label, origin tex (relative to repo root),
# generator script, data-producing script, and optional per-block suffixes
# (when one .tex holds several \begin{tabular} blocks).
TABLES = [
    {"name": "Table18", "label": "tab:corpus-provenance",
     "origin": "4_outputs/mpnet/tables/tab18_corpus_provenance.tex",
     "generator": "1_code/7_main_analysis/2_appendix/export_corpus_provenance.py",
     "data": "1_code/2_segment (hydrated snapshot counts)"},
    {"name": "Table16", "label": "tab:model-selection-ranking",
     "origin": "4_outputs/mpnet/tables/tab16_model_selection_ranking.tex",
     "generator": "1_code/7_main_analysis/2_appendix/d1_export_model_selection_nums.py",
     "data": "1_code/4_supervised_model_train/1_grid_search.py"},
    {"name": "Table1", "label": "tab:validation",
     "origin": "4_outputs/mpnet/tables/tab1_classifier_performance.tex",
     "generator": "1_code/7_main_analysis/1_main_text/3_generate_cross_sensitivity_table.py",
     "data": "1_code/5_supervised_model_infer/score_supervised.py"},
    {"name": "Table5", "label": "tab:register-decomposition",
     "origin": "4_outputs/mpnet/tables/tab5_register_decomposition.tex",
     "generator": "1_code/7_main_analysis/0_shared/g_register_decomposition.py",
     "data": "1_code/7_main_analysis/1_main_text/1_semantic_gap.py"},
    {"name": "Table4", "label": "tab:interaction",
     "origin": "4_outputs/mpnet/tables/tab4_interaction_h25.tex",
     "generator": "1_code/7_main_analysis/1_main_text/2_coverage_semantic_interaction.py",
     "data": "1_code/5_supervised_model_infer/score_supervised.py"},
    {"name": "Table10", "label": "tab:concept-coverage",
     "origin": "4_outputs/mpnet/tables/tab10_concept_coverage.tex",
     "generator": "1_code/7_main_analysis/1_main_text/3_generate_cross_sensitivity_table.py",
     "data": "1_code/7_main_analysis/1_main_text/1_semantic_gap.py"},
    {"name": "Table11", "label": "tab:reference-concept",
     "origin": "4_outputs/mpnet/tables/tab_concept_reference.tex",
     "generator": "1_code/7_main_analysis/0_shared/g_register_decomposition.py",
     "data": "1_code/7_main_analysis/1_main_text/1_semantic_gap.py"},
    {"name": "TableA3", "label": "tab:sdg4-lexical-audit",
     "origin": "4_outputs/appendix/mpnet/a3_sdg4_audit/tables/tab_a3_sdg4_lexical_audit.tex",
     "generator": "1_code/7_main_analysis/2_appendix/a3_sdg4_lexical_audit.py",
     "data": "1_code/5_supervised_model_infer/score_supervised.py"},
    {"name": "TableB2", "label": "tab:semantic-gap-text-interpretability-full",
     "origin": "4_outputs/appendix/mpnet/b2_semantic_gap_interpretability/tables/tab_b2_semantic_gap_interpret_all.tex",
     "generator": "1_code/7_main_analysis/2_appendix/b2_semantic_gap_text_interpretability.py",
     "data": "1_code/7_main_analysis/1_main_text/1_semantic_gap.py"},
    {"name": "Table13", "label": "tab:distributional-gap", "parts": ["", "_cont"],
     "origin": "4_outputs/mpnet/adjusted/tables/tab13_distributional_gap.tex",
     "generator": "1_code/7_main_analysis/1_main_text/g_distributional_gap.py",
     "data": "1_code/5_supervised_model_infer/score_supervised.py"},
    {"name": "Table14", "label": "tab:distributional-h1",
     "origin": "4_outputs/mpnet/adjusted/tables/tab14_distributional_h1.tex",
     "generator": "1_code/7_main_analysis/1_main_text/g_distributional_h1_correlation.py",
     "data": "1_code/7_main_analysis/1_main_text/g_distributional_gap.py"},
    {"name": "TableC", "label": "tab:sample-stability",
     "origin": "4_outputs/appendix/mpnet/c_sample_stability/tables/tab_c_sample_stability.tex",
     "generator": "1_code/7_main_analysis/2_appendix/c_sample_stability.py",
     "data": "1_code/5_supervised_model_infer/score_supervised.py"},
    {"name": "Table16_full", "label": "tab:model-selection-appendix",
     "origin": "4_outputs/mpnet/tables/tab16_model_selection_full.tex",
     "generator": "1_code/7_main_analysis/2_appendix/d1_export_model_selection_nums.py",
     "data": "1_code/4_supervised_model_train/1_grid_search.py"},
    {"name": "Table12_cross", "label": "tab:iterative-register-check",
     "origin": "4_outputs/mpnet/tables/tab12_register_cross.tex",
     "generator": "1_code/7_main_analysis/2_appendix/f3_register_iterative_cross_table.py",
     "data": "1_code/7_main_analysis/0_shared/register_adjust.py"},
    {"name": "TableA1_design", "label": "tab:register-sample-design",
     "origin": "4_outputs/appendix/mpnet/a1_register_validation/tables/tab_a1_register_sample_design.tex",
     "generator": "1_code/7_main_analysis/2_appendix/a1_register_validation.py",
     "data": "1_code/7_main_analysis/0_shared/register_adjust.py"},
    {"name": "TableA1_features", "label": "tab:register-feature-contrasts",
     "origin": "4_outputs/appendix/mpnet/a1_register_validation/tables/tab_a1_register_features.tex",
     "generator": "1_code/7_main_analysis/2_appendix/a1_register_validation.py",
     "data": "1_code/7_main_analysis/0_shared/register_adjust.py"},
    {"name": "TableA1", "label": "tab:register-validation-accuracy",
     "origin": "4_outputs/appendix/mpnet/a1_register_validation/tables/tab_a1_register_validation.tex",
     "generator": "1_code/7_main_analysis/2_appendix/a1_register_validation.py",
     "data": "1_code/7_main_analysis/0_shared/register_adjust.py"},
    {"name": "TableA1_selectivity", "label": "tab:register-validation-selectivity",
     "origin": "4_outputs/appendix/mpnet/a1_register_validation/tables/tab_a1_register_validation_selectivity.tex",
     "generator": "1_code/7_main_analysis/2_appendix/a1_register_validation.py",
     "data": "1_code/7_main_analysis/0_shared/register_adjust.py"},
    {"name": "Table6a", "label": "tab:cross-sensitivity-robustness",
     "origin": "4_outputs/mpnet/tables/tab6a_cross_sensitivity.tex",
     "generator": "1_code/7_main_analysis/1_main_text/3_generate_cross_sensitivity_table.py",
     "data": "1_code/5_supervised_model_infer/score_supervised.py"},
    {"name": "Table6b", "label": "tab:cross-sensitivity-robustness-raw",
     "origin": "4_outputs/mpnet/tables/tab6b_cross_sensitivity.tex",
     "generator": "1_code/7_main_analysis/1_main_text/3_generate_cross_sensitivity_table.py",
     "data": "1_code/5_supervised_model_infer/score_supervised.py"},
    {"name": "Table7a", "label": "tab:encoder-sensitivity-semantic",
     "origin": "4_outputs/mpnet/tables/tab7a_encoder_sensitivity.tex",
     "generator": "1_code/7_main_analysis/1_main_text/3_generate_cross_sensitivity_table.py",
     "data": "1_code/5_supervised_model_infer/score_supervised.py"},
    {"name": "Table7b", "label": "tab:encoder-sensitivity-semantic-raw",
     "origin": "4_outputs/mpnet/tables/tab7b_encoder_sensitivity.tex",
     "generator": "1_code/7_main_analysis/1_main_text/3_generate_cross_sensitivity_table.py",
     "data": "1_code/5_supervised_model_infer/score_supervised.py"},
    {"name": "Table9", "label": "tab:encoder-sensitivity-coverage",
     "origin": "4_outputs/mpnet/tables/tab9_encoder_sensitivity_coverage.tex",
     "generator": "1_code/7_main_analysis/1_main_text/3_generate_cross_sensitivity_table.py",
     "data": "1_code/5_supervised_model_infer/score_supervised.py"},
    {"name": "Table8", "label": "tab:cross-sensitivity-coverage",
     "origin": "4_outputs/mpnet/tables/tab8_coverage_sensitivity.tex",
     "generator": "1_code/7_main_analysis/1_main_text/3_generate_cross_sensitivity_table.py",
     "data": "1_code/7_main_analysis/1_main_text/2_coverage_semantic_interaction.py"},
    {"name": "Table_minilm_ref", "label": "tab:reference-minilm",
     "origin": "4_outputs/minilm/tables/tab_minilm_reference.tex",
     "generator": "1_code/7_main_analysis/0_shared/g_register_decomposition.py",
     "data": "1_code/7_main_analysis/1_main_text/1_semantic_gap.py"},
    {"name": "Table_scibert_ref", "label": "tab:reference-scibert",
     "origin": "4_outputs/scibert/tables/tab_scibert_reference.tex",
     "generator": "1_code/7_main_analysis/0_shared/g_register_decomposition.py",
     "data": "1_code/7_main_analysis/1_main_text/1_semantic_gap.py"},
    {"name": "TableH1_covgap", "label": "tab:app-cross-method-covgap",
     "origin": "4_outputs/appendix/mpnet/h1_cross_method_gap_values/tables/tab_app_cross_method_covgap.tex",
     "generator": "1_code/7_main_analysis/2_appendix/h1_cross_method_gap_values.py",
     "data": "1_code/7_main_analysis/1_main_text/2_coverage_semantic_interaction.py"},
    {"name": "TableH1_semgap", "label": "tab:app-cross-method-semgap",
     "origin": "4_outputs/appendix/mpnet/h1_cross_method_gap_values/tables/tab_app_cross_method_semgap.tex",
     "generator": "1_code/7_main_analysis/2_appendix/h1_cross_method_gap_values.py",
     "data": "1_code/7_main_analysis/1_main_text/1_semantic_gap.py"},
    {"name": "TableI1", "label": "tab:app-assignment-method-comparison",
     "origin": "4_outputs/appendix/mpnet/i1_assignment_method_comparison/tables/tab_app_assignment_method_comparison.tex",
     "generator": "1_code/7_main_analysis/2_appendix/i1_assignment_method_comparison.py",
     "data": "1_code/5_supervised_model_infer/score_supervised.py"},
    {"name": "TableJ1", "label": "tab:raw-value-correlation",
     "origin": "4_outputs/appendix/mpnet/j1_raw_value_correlation/tables/tab_j1_raw_value_correlation.tex",
     "generator": "1_code/7_main_analysis/2_appendix/j1_raw_value_correlation.py",
     "data": "1_code/7_main_analysis/1_main_text/2_coverage_semantic_interaction.py"},
    {"name": "TableA2", "label": "tab:policy-source-family",
     "origin": "4_outputs/appendix/mpnet/a2_source_family_sensitivity/tables/tab_a2_policy_source_family_combined.tex",
     "generator": "1_code/7_main_analysis/2_appendix/a2_policy_source_family_sensitivity.py",
     "data": "1_code/7_main_analysis/1_main_text/2_coverage_semantic_interaction.py"},
    {"name": "TableA2_h25", "label": "tab:policy-source-family-h25",
     "origin": "4_outputs/appendix/mpnet/a2_source_family_sensitivity/tables/tab_a2_policy_source_family_h25.tex",
     "generator": "1_code/7_main_analysis/2_appendix/a2_policy_source_family_sensitivity.py",
     "data": "1_code/7_main_analysis/1_main_text/2_coverage_semantic_interaction.py"},
    {"name": "TableK1", "label": "tab:k1-specification-grid",
     "origin": "4_outputs/appendix/mpnet/k1_regression_semantic_gap/tables/tab_k1_specification_grid.tex",
     "generator": "1_code/7_main_analysis/2_appendix/k1_regression_semantic_gap.py",
     "data": "1_code/7_main_analysis/1_main_text/1_semantic_gap.py"},
]

# Figures by PRINTED PDF order. Each "sources" entry is (origin relative path,
# final filename).
FIGURES = [
    {"name": "Fig1", "label": "fig:conceptual-framework",
     "generator": "1_code/8_visualization/build_conceptual_figs.py", "data": "same",
     "sources": [("4_outputs/conceptual_figs/fig1_conceptual_framework.png", "Fig1.png"),
                 ("4_outputs/conceptual_figs/fig1_conceptual_framework.pdf", "Fig1.pdf")]},
    {"name": "Fig2", "label": "fig:pipeline-flowchart",
     "generator": "1_code/8_visualization/build_pipeline_flowchart.py", "data": "same",
     "sources": [("4_outputs/conceptual_figs/fig6_pipeline_flowchart.png", "Fig2.png"),
                 ("4_outputs/conceptual_figs/fig6_pipeline_flowchart.pdf", "Fig2.pdf")]},
    {"name": "Fig3", "label": "fig:coverage_profiles",
     "generator": "1_code/8_visualization/plot_figures.py",
     "data": "1_code/7_main_analysis/1_main_text/2_coverage_semantic_interaction.py",
     "sources": [("4_outputs/mpnet/figures/fig2_coverage_profiles.png", "Fig3.png"),
                 ("4_outputs/mpnet/figures/fig2_coverage_profiles.pdf", "Fig3.pdf")]},
    {"name": "Fig4", "label": "fig:pca-register-before-after",
     "generator": "1_code/7_main_analysis/1_main_text/0_pca_register_before_after.py",
     "data": "1_code/7_main_analysis/0_shared/register_adjust.py",
     "sources": [("4_outputs/mpnet/figures/fig3_pca_register_before_after.png", "Fig4.png"),
                 ("4_outputs/mpnet/figures/fig3_pca_register_before_after.pdf", "Fig4.pdf")]},
    {"name": "Fig5", "label": "fig:semantic_gap",
     "generator": "1_code/8_visualization/plot_figures.py",
     "data": "1_code/7_main_analysis/1_main_text/1_semantic_gap.py",
     "sources": [("4_outputs/mpnet/figures/fig4_semantic_gap.png", "Fig5.png"),
                 ("4_outputs/mpnet/figures/fig4_semantic_gap.pdf", "Fig5.pdf")]},
    {"name": "Fig6", "label": "fig:typology_scatter",
     "generator": "1_code/8_visualization/plot_figures.py",
     "data": "1_code/7_main_analysis/1_main_text/2_coverage_semantic_interaction.py",
     "sources": [("4_outputs/mpnet/figures/fig9_h1a_scatter.png", "Fig6a.png"),
                 ("4_outputs/mpnet/figures/fig9_h1a_scatter.pdf", "Fig6a.pdf"),
                 ("4_outputs/mpnet/figures/fig9_h1b_scatter.png", "Fig6b.png"),
                 ("4_outputs/mpnet/figures/fig9_h1b_scatter.pdf", "Fig6b.pdf"),
                 ("4_outputs/mpnet/figures/fig9_h1c_scatter.png", "Fig6c.png"),
                 ("4_outputs/mpnet/figures/fig9_h1c_scatter.pdf", "Fig6c.pdf"),
                 ("4_outputs/mpnet/figures/fig9_h1d_scatter.png", "Fig6d.png"),
                 ("4_outputs/mpnet/figures/fig9_h1d_scatter.pdf", "Fig6d.pdf")]},
    {"name": "Fig7", "label": "fig:centroid-similarity-matrix",
     "generator": "1_code/8_visualization/plot_figures.py",
     "data": "1_code/6_calculate_centroids/1_build_centroid_similarity_matrix.py",
     "sources": [("4_outputs/appendix/mpnet/a4_centroid_similarity/figures/fig8_centroid_similarity_heatmap.png", "Fig7.png"),
                 ("4_outputs/appendix/mpnet/a4_centroid_similarity/figures/fig8_centroid_similarity_heatmap.pdf", "Fig7.pdf")]},
]

_SYMBOL_MAP = {
    r"\rho": "ρ", r"\pm": "±", r"\geq": "≥", r"\leq": "≤",
    r"\times": "×", r"\approx": "≈", r"\dots": "…", r"\S": "§",
    r"\alpha": "α", r"\beta": "β", r"\sigma": "σ", r"\mu": "μ",
    r"\gamma": "γ", r"\delta": "δ", r"\lambda": "λ",
    r"\rightarrow": "→", r"\leftarrow": "←",
}

# Alignment column may itself contain braces, e.g. p{0.28\textwidth}.
_MULTICOLUMN_RE = re.compile(r"\\multicolumn\{(\d+)\}\{(?:[^{}]|\{[^{}]*\})*\}\{([^{}]*)\}")
_MULTIROW_RE = re.compile(r"\\multirow\{[^}]*\}\{[^}]*\}\{([^{}]*)\}")
_RULE_RE = re.compile(r"\\(toprule|midrule|bottomrule|hline|cmidrule|addlinespace|specialrule)")
# Strips a whole rule command plus its optional (lr) / {1-2} arguments.
_RULE_STRIP_RE = re.compile(
    r"\\(toprule|midrule|bottomrule|hline|cmidrule|addlinespace|specialrule)"
    r"(?:\([^)]*\))?(?:\s*\{[^}]*\})?"
)
# \shortstack{Coverage gap\\(vs policy)} -> "Coverage gap (vs policy)"; its internal
# \\ would otherwise be mistaken for a row separator.
_SHORTSTACK_RE = re.compile(r"\\shortstack\{([^}]*)\}")


def _expand_multicolumn(row: str) -> str:
    def _repl(m: re.Match) -> str:
        n = int(m.group(1))
        return m.group(2) + ("\u0001" * (n - 1))

    return _MULTICOLUMN_RE.sub(_repl, row)


def _clean_cell(cell: str) -> str:
    s = cell.strip()
    s = s.replace("\\%", "%").replace("\\&", "&").replace("\\#", "#")
    for cmd in ("textbf", "emph", "texttt", "textit", "textsc", "uline"):
        s = re.sub(r"\\" + cmd + r"\{([^{}]*)\}", r"\1", s)
    s = _MULTIROW_RE.sub(r"\1", s)
    s = _MULTICOLUMN_RE.sub(r"\2", s)  # safety net for any unexpanded multicolumn
    for k, v in _SYMBOL_MAP.items():
        s = s.replace(k, v)
    # remaining spacing / formatting commands
    s = re.sub(r"\\[,;:! ]", " ", s)
    s = s.replace("\\textwidth", "")
    s = re.sub(r"\\[a-zA-Z]+", "", s)  # drop any other stray control word
    s = s.replace("$", "").replace("{", "").replace("}", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _tex_to_tables(path: Path) -> list[list[list[str]]]:
    """Return a list of tables; each table is a list of rows; each row is a list of cells."""
    text = path.read_text(encoding="utf-8")
    # The column spec may contain nested braces, e.g. p{0.28\textwidth}; use a
    # brace-balanced matcher so the spec is consumed whole and not leaked into cells.
    spec = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
    block_re = re.compile(
        r"\\begin\{tabularx?\*?\}" + spec + r"(.*?)\\end\{tabularx?\*?\}", re.DOTALL
    )
    blocks = block_re.findall(text)
    out = []
    for blk in blocks:
        blk = _SHORTSTACK_RE.sub(lambda m: m.group(1).replace("\\\\", " "), blk)
        rows = []
        for raw in re.split(r"\\\\\s*(?:\*|\[[^\]]*\])?", blk):
            r = _RULE_STRIP_RE.sub("", raw).strip()
            if not r:
                continue
            expanded = _expand_multicolumn(r)
            cells = [_clean_cell(c) for c in re.split(r"(?<!\\)&|\u0001", expanded)]
            rows.append(cells)
        if rows:
            out.append(rows)
    return out


def _write_xlsx(rows: list[list[str]], dest: Path) -> None:
    max_cols = max((len(r) for r in rows), default=0)
    padded = [r + [""] * (max_cols - len(r)) for r in rows]
    df = pd.DataFrame(padded)
    with pd.ExcelWriter(dest, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Sheet1", index=False, header=False)
        ws = writer.sheets["Sheet1"]
        for col in ws.columns:
            width = max((len(str(c.value)) for c in col if c.value is not None), default=10)
            ws.column_dimensions[col[0].column_letter].width = min(width + 2, 60)
        for cell in ws[1]:
            cell.font = Font(bold=True)


def collect_final_outputs(output_dir: Path, model: str = "all-mpnet-base-v2") -> None:
    output_dir = Path(output_dir)
    final_dir = output_dir / "final"
    tex_dir = final_dir / "tables_tex"
    final_dir.mkdir(parents=True, exist_ok=True)
    tex_dir.mkdir(parents=True, exist_ok=True)

    readme_lines = ["# 4_outputs/final — human-facing bundle (MPNet track)\n",
                    "Generated by `python main.py --get-outputs-final`.\n"]

    # ----- tables -----
    readme_lines.append("\n## Tables (one .xlsx each; original .tex in tables_tex/)\n")
    for t in TABLES:
        origin = REPO_ROOT / t["origin"]
        if not origin.exists():
            print(f"[skip] missing {origin}", file=sys.stderr)
            readme_lines.append(f"- **{t['name']}** — MISSING: {t['origin']}\n")
            continue
        parts = t.get("parts", [""])
        tables = _tex_to_tables(origin)
        if not tables:
            print(f"[warn] no tabular parsed in {origin}", file=sys.stderr)
        for i, rows in enumerate(tables):
            suffix = parts[i] if i < len(parts) else f"_{i+1}"
            dest = final_dir / f"{t['name']}{suffix}.xlsx"
            _write_xlsx(rows, dest)
        shutil.copy2(origin, tex_dir / origin.name)
        print(f"# ----- {t['name']} ------")
        print(f"# output_origin = {t['origin']}")
        print(f"# output_final  = 4_outputs/final/{t['name']}.xlsx (tex->xlsx conversion)")
        print(f"# generator script : {t['generator']}")
        print(f"# data script      : {t['data']}")
        readme_lines.append(
            f"- **{t['name']}** (`{t['name']}.xlsx`) — origin `{t['origin']}`  \n"
            f"  generator: `{t['generator']}`  \n"
            f"  data: `{t['data']}`\n"
        )

    # ----- figures -----
    readme_lines.append("\n## Figures (copied as FigX.png / FigX.pdf)\n")
    for f in FIGURES:
        print(f"# ----- {f['name']} ------")
        print(f"# generator script : {f['generator']}")
        print(f"# data script      : {f['data']}")
        for src_rel, dest_name in f["sources"]:
            src = REPO_ROOT / src_rel
            dest = final_dir / dest_name
            if not src.exists():
                print(f"[skip] missing {src}", file=sys.stderr)
                readme_lines.append(f"- **{dest_name}** — MISSING: {src_rel}\n")
                continue
            shutil.copy2(src, dest)
            print(f"# output_final  = 4_outputs/final/{dest_name}  (copied from {src_rel})")
        readme_lines.append(
            f"- **{f['name']}** (`{f['name']}.*`) — generator `{f['generator']}`  \n"
            f"  data: `{f['data']}`\n"
        )

    (final_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")
    print(f"\nDone. Bundle written to {final_dir}")


if __name__ == "__main__":
    out = REPO_ROOT / "4_outputs"
    collect_final_outputs(out)
