"""Run the preregistered qualification for two independent experiments."""

from calibration.overlap_campaign import main
from increment.overlap import _evaluate_overlap_draw

if __name__ == "__main__":
    raise SystemExit(main(_evaluate_overlap_draw))
