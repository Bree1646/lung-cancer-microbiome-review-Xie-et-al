"""Rebuilds every figure, table and number into ../results/."""
import figure1
import figure2
import figure3
import figure4
import figureS1
import figureS2
import figureS3
import figureS4
import figureS5
import manuscript_numbers

for module in (figure1, figure2, figure3, figure4, figureS1, figureS2, figureS3, figureS4, figureS5,
               manuscript_numbers):
    module.main()
