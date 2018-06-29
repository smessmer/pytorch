#include "caffe2/operators/generate_proposals_op_util_boxes.h"
#include "caffe2/utils/math_eigen.h"

#include <gtest/gtest.h>

namespace caffe2 {

TEST(UtilsBoxesTest, TestBboxTransformRandom) {
  using EMatXf = Eigen::MatrixXf;

  EMatXf bbox(5, 4);
  bbox << 175.62031555, 20.91103172, 253.352005, 155.0145874, 169.24636841,
      4.85241556, 228.8605957, 105.02092743, 181.77426147, 199.82876587,
      192.88427734, 214.0255127, 174.36262512, 186.75761414, 296.19091797,
      231.27906799, 22.73153877, 92.02596283, 135.5695343, 208.80291748;

  EMatXf deltas(5, 4);
  deltas << 0.47861834, 0.13992102, 0.14961673, 0.71495209, 0.29915856,
      -0.35664671, 0.89018666, 0.70815367, -0.03852064, 0.44466892, 0.49492538,
      0.71409376, 0.28052918, 0.02184832, 0.65289006, 1.05060139, -0.38172557,
      -0.08533806, -0.60335309, 0.79052375;

  EMatXf result_gt(5, 4);
  result_gt << 206.94953073, -30.71519157, 298.3876512, 245.44846569,
      143.8712194, -83.34289038, 291.50227513, 122.05339902, 177.43029521,
      198.66623633, 197.29527254, 229.70308414, 152.25190373, 145.43156421,
      388.21547899, 275.59425266, 5.06242193, 11.04094661, 67.32890274,
      270.68622005;

  const float BBOX_XFORM_CLIP = log(1000.0 / 16.0);
  auto result = utils::bbox_transform(
      bbox.array(),
      deltas.array(),
      std::vector<float>{1.0, 1.0, 1.0, 1.0},
      BBOX_XFORM_CLIP);
  EXPECT_NEAR((result.matrix() - result_gt).norm(), 0.0, 1e-4);
}

} // namespace caffe2
