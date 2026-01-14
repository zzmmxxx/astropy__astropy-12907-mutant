import numpy as np
from numpy.testing import assert_allclose
from astropy.modeling import models
from astropy.modeling.models import Mapping
from astropy.modeling.separable import (_coord_matrix, is_separable, separability_matrix)

print('=' * 60)
print('运行 astropy separable 测试')
print('=' * 60)

# 创建测试模型
sh1 = models.Shift(1, name='shift1')
sh2 = models.Shift(2, name='sh2')
scl1 = models.Scale(1, name='scl1')
scl2 = models.Scale(2, name='scl2')
map1 = Mapping((0, 1, 0, 1), name='map1')
map2 = Mapping((0, 0, 1), name='map2')
map3 = Mapping((0, 0), name='map3')
rot = models.Rotation2D(2, name='rotation')
p2 = models.Polynomial2D(1, name='p2')
p22 = models.Polynomial2D(2, name='p22')
p1 = models.Polynomial1D(1, name='p1')

# 定义compound_models
compound_models = {
    'cm1': (map3 & sh1 | rot & sh1 | sh1 & sh2 & sh1,
            (np.array([False, False, True]),
             np.array([[True, False], [True, False], [False, True]]))
            ),
    'cm2': (sh1 & sh2 | rot | map1 | p2 & p22,
            (np.array([False, False]),
             np.array([[True, True], [True, True]]))
            ),
    'cm3': (map2 | rot & scl1,
            (np.array([False, False, True]),
             np.array([[True, False], [True, False], [False, True]]))
            ),
    'cm4': (sh1 & sh2 | map2 | rot & scl1,
            (np.array([False, False, True]),
             np.array([[True, False], [True, False], [False, True]]))
            ),
    'cm5': (map3 | sh1 & sh2 | scl1 & scl2,
            (np.array([False, False]),
             np.array([[True], [True]]))
            ),
    'cm7': (map2 | p2 & sh1,
            (np.array([False, True]),
             np.array([[True, False], [False, True]]))
            )
}

print('测试 test_coord_matrix:')
try:
    c = _coord_matrix(p2, 'left', 2)
    assert_allclose(np.array([[1, 1], [0, 0]]), c)
    print('✓ test_coord_matrix: PASSED')
    coord_passed = True
except Exception as e:
    print(f'✗ test_coord_matrix: FAILED - {e}')
    coord_passed = False

print()
print('测试 separable 函数:')
passed = 0
failed = 0

for i, (name, (compound_model, expected)) in enumerate(compound_models.items()):
    test_name = f'compound_model{i}'
    try:
        actual_sep = is_separable(compound_model)
        actual_mat = separability_matrix(compound_model)
        assert_allclose(actual_sep, expected[0])
        assert_allclose(actual_mat, expected[1])
        print(f'✓ {test_name} ({name}): PASSED')
        passed += 1
    except Exception as e:
        print(f'✗ {test_name} ({name}): FAILED - {e}')
        failed += 1

print()
print('=' * 60)
print('测试总结:')
print(f'test_coord_matrix: {"PASSED" if coord_passed else "FAILED"}')
print(f'separable 测试通过: {passed}')
print(f'separable 测试失败: {failed}')
print(f'总计: {passed + failed + 1} 个测试')

print()
print('关于用户提到的测试用例:')
print(f'- compound_model6: 不存在 (只有 {len(compound_models)} 个模型)')
print(f'- compound_model9: 不存在 (只有 {len(compound_models)} 个模型)')
print('可用的测试用例索引: 0-5')