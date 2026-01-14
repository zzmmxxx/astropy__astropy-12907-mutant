#!/usr/bin/env python
"""
使用系统安装的astropy来运行separable测试
"""
import sys
import numpy as np
from numpy.testing import assert_allclose

def run_system_tests():
    """使用系统astropy运行测试"""
    print("=" * 60)
    print("使用系统 astropy 运行 separable 测试")
    print("=" * 60)
    
    try:
        # 导入系统astropy
        from astropy.modeling import models
        from astropy.modeling.models import Mapping
        from astropy.modeling.separable import (_coord_matrix, is_separable, _cdot,
                                                _cstack, _arith_oper, separability_matrix)
        print("✓ 成功导入系统 astropy.modeling")
        
        # 创建测试模型 (与test_separable.py相同)
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
        print("✓ 成功创建测试模型")
        
        # 定义compound_models (与test_separable.py相同)
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
        
        print(f"✓ 定义了 {len(compound_models)} 个复合模型")
        
        # 运行test_coord_matrix测试
        print("\n" + "=" * 40)
        print("运行 test_coord_matrix 测试")
        print("=" * 40)
        
        coord_matrix_passed = True
        try:
            c = _coord_matrix(p2, 'left', 2)
            assert_allclose(np.array([[1, 1], [0, 0]]), c)
            c = _coord_matrix(p2, 'right', 2)
            assert_allclose(np.array([[0, 0], [1, 1]]), c)
            c = _coord_matrix(p1, 'left', 2)
            assert_allclose(np.array([[1], [0]]), c)
            c = _coord_matrix(p1, 'left', 1)
            assert_allclose(np.array([[1]]), c)
            c = _coord_matrix(sh1, 'left', 2)
            assert_allclose(np.array([[1], [0]]), c)
            c = _coord_matrix(sh1, 'right', 2)
            assert_allclose(np.array([[0], [1]]), c)
            c = _coord_matrix(sh1, 'right', 3)
            assert_allclose(np.array([[0], [0], [1]]), c)
            c = _coord_matrix(map3, 'left', 2)
            assert_allclose(np.array([[1], [1]]), c)
            c = _coord_matrix(map3, 'left', 3)
            assert_allclose(np.array([[1], [1], [0]]), c)
            
            print("✓ test_coord_matrix: PASSED")
            
        except Exception as e:
            print(f"✗ test_coord_matrix: FAILED - {e}")
            coord_matrix_passed = False
        
        # 运行separable测试
        print("\n" + "=" * 40)
        print("运行 test_separable 测试")
        print("=" * 40)
        
        passed_tests = []
        failed_tests = []
        
        for i, (name, (compound_model, expected_result)) in enumerate(compound_models.items()):
            test_name = f"compound_model{i}"
            print(f"\n测试 {test_name} ({name}):")
            
            try:
                actual_separable = is_separable(compound_model)
                actual_matrix = separability_matrix(compound_model)
                
                print(f"  预期 separable: {expected_result[0]}")
                print(f"  实际 separable: {actual_separable}")
                print(f"  预期 matrix shape: {expected_result[1].shape}")
                print(f"  实际 matrix shape: {actual_matrix.shape}")
                
                assert_allclose(actual_separable, expected_result[0])
                assert_allclose(actual_matrix, expected_result[1])
                
                print(f"✓ {test_name}: PASSED")
                passed_tests.append(test_name)
                
            except Exception as e:
                print(f"✗ {test_name}: FAILED - {e}")
                print(f"  预期 separable: {expected_result[0]}")
                print(f"  实际 separable: {actual_separable if 'actual_separable' in locals() else 'N/A'}")
                print(f"  预期 matrix:\n{expected_result[1]}")
                print(f"  实际 matrix:\n{actual_matrix if 'actual_matrix' in locals() else 'N/A'}")
                failed_tests.append(test_name)
        
        # 检查用户提到的测试用例
        print(f"\n" + "=" * 40)
        print("检查用户提到的FAIL_TO_PASS测试用例")
        print("=" * 40)
        
        print(f"用户提到的测试:")
        print(f"- compound_model6: 不存在 (只有 {len(compound_models)} 个模型, 索引 0-{len(compound_models)-1})")
        print(f"- compound_model9: 不存在 (只有 {len(compound_models)} 个模型, 索引 0-{len(compound_models)-1})")
        
        print(f"\n可用的测试用例:")
        for i, name in enumerate(compound_models.keys()):
            status = "PASSED" if f"compound_model{i}" in passed_tests else "FAILED"
            print(f"- compound_model{i} ({name}): {status}")
        
        print(f"\n" + "=" * 60)
        print("测试总结")
        print("=" * 60)
        print(f"test_coord_matrix: {'PASSED' if coord_matrix_passed else 'FAILED'}")
        print(f"通过的 separable 测试: {len(passed_tests)}")
        print(f"失败的 separable 测试: {len(failed_tests)}")
        print(f"总测试数: {len(passed_tests) + len(failed_tests) + 1}")
        
        if failed_tests:
            print(f"\n失败的测试: {', '.join(failed_tests)}")
        
        return len(failed_tests) == 0 and coord_matrix_passed
        
    except Exception as e:
        print(f"运行测试时发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_system_tests()
    print(f"\n{'=' * 60}")
    print(f"测试结果: {'SUCCESS' if success else 'FAILURE'}")
    print(f"{'=' * 60}")
    sys.exit(0 if success else 1)