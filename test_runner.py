#!/usr/bin/env python
"""
简化的测试运行器，用于测试separable模块
"""
import sys
import os
import traceback

# 添加当前目录到Python路径
sys.path.insert(0, '.')

def run_tests():
    """运行测试"""
    print("=" * 60)
    print("开始运行 astropy separable 测试")
    print("=" * 60)
    
    try:
        # 尝试导入必要的模块
        print("正在导入依赖模块...")
        import numpy as np
        from numpy.testing import assert_allclose
        print("✓ numpy 导入成功")
        
        # 尝试导入astropy模块（可能会失败）
        try:
            from astropy.modeling import models
            from astropy.modeling.models import Mapping
            from astropy.modeling.separable import (_coord_matrix, is_separable, _cdot,
                                                    _cstack, _arith_oper, separability_matrix)
            print("✓ astropy.modeling 模块导入成功")
            astropy_available = True
        except Exception as e:
            print(f"✗ astropy.modeling 模块导入失败: {e}")
            astropy_available = False
            
        if not astropy_available:
            print("\n由于astropy模块导入失败，无法运行完整测试")
            print("这通常是因为扩展模块未构建")
            return False
            
        # 创建测试模型
        print("\n正在创建测试模型...")
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
        print("✓ 测试模型创建成功")
        
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
        
        print(f"\n定义了 {len(compound_models)} 个复合模型")
        
        # 运行test_coord_matrix测试
        print("\n" + "=" * 40)
        print("运行 test_coord_matrix 测试")
        print("=" * 40)
        
        try:
            # test_coord_matrix的内容
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
            traceback.print_exc()
        
        # 运行separable测试
        print("\n" + "=" * 40)
        print("运行 test_separable 测试")
        print("=" * 40)
        
        passed_tests = 0
        failed_tests = 0
        
        for i, (name, (compound_model, result)) in enumerate(compound_models.items()):
            test_name = f"compound_model{i}"
            print(f"\n测试 {test_name} ({name}):")
            
            try:
                actual_separable = is_separable(compound_model)
                actual_matrix = separability_matrix(compound_model)
                
                assert_allclose(actual_separable, result[0])
                assert_allclose(actual_matrix, result[1])
                
                print(f"✓ {test_name}: PASSED")
                passed_tests += 1
                
            except Exception as e:
                print(f"✗ {test_name}: FAILED - {e}")
                print(f"  预期 separable: {result[0]}")
                print(f"  实际 separable: {actual_separable if 'actual_separable' in locals() else 'N/A'}")
                print(f"  预期 matrix: {result[1]}")
                print(f"  实际 matrix: {actual_matrix if 'actual_matrix' in locals() else 'N/A'}")
                failed_tests += 1
        
        # 测试用户提到的compound_model6和compound_model9
        print(f"\n" + "=" * 40)
        print("检查用户提到的测试用例")
        print("=" * 40)
        
        print(f"compound_model6: 不存在 (只有 {len(compound_models)} 个模型, 索引 0-{len(compound_models)-1})")
        print(f"compound_model9: 不存在 (只有 {len(compound_models)} 个模型, 索引 0-{len(compound_models)-1})")
        
        print(f"\n" + "=" * 60)
        print("测试总结")
        print("=" * 60)
        print(f"通过的测试: {passed_tests}")
        print(f"失败的测试: {failed_tests}")
        print(f"总测试数: {passed_tests + failed_tests + 1}")  # +1 for test_coord_matrix
        
        return failed_tests == 0
        
    except Exception as e:
        print(f"运行测试时发生错误: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)