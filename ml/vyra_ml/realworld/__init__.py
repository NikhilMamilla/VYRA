"""Phase 3A: real-world validation of the synthetic-trained baseline.

This subpackage is an *experiment*. It never trains or tunes anything against
real data -- it loads the frozen Phase 2 model, runs the identical feature
pipeline on VizWiz-QualityIssues images, and measures transfer.
"""
