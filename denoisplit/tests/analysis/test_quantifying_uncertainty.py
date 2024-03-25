# from denoisplit.analysis.quantifying_uncertainty import compute_regionwise_metric_one_pair, aggregate_metric
# import numpy as np
#
#
# def equal(a, b, eps=1e-7):
#     return np.abs(a - b) < eps
#
#
# def test_compute_regionwise_metric_one_pair_with_no_region():
#     data1 = np.random.random((1, 4, 4))
#     data2 = data1.copy()
#     regionsize = 1
#     data2[0, 1, 1] += 1
#     data2[0, 3, 3] += 5
#     output = compute_regionwise_metric_one_pair(data1, data2, ['RMSE'], regionsize)
#     assert output['RMSE'].shape == data1.shape
#     for i in range(4):
#         for j in range(4):
#             val = output['RMSE'][0, i, j]
#             if i == 1 and j == 1:
#                 assert equal(val, 1)
#             elif i == 3 and j == 3:
#                 assert equal(val, 5)
#             else:
#                 assert equal(val, 0)
#
#
# def test_compute_regionwise_metric_one_pair():
#     """
#     tests for a regionsize of 2*2
#     """
#     data1 = np.random.random((1, 4, 4))
#     data2 = data1.copy()
#     regionsize = 2
#     data2[0, 1, 1] += 3
#     data2[0, 0, 0] += 4
#
#     data2[0, 2, 3] += 12
#     data2[0, 3, 2] += 5
#
#     output = compute_regionwise_metric_one_pair(data1, data2, ['RMSE'], regionsize)
#     assert output['RMSE'].shape == (1, 2, 2)
#     assert equal(output['RMSE'][0, 0, 0], 2.5)
#     assert equal(output['RMSE'][0, 1, 1], 6.5)
#     assert equal(output['RMSE'][0, 0, 1], 0)
#     assert equal(output['RMSE'][0, 1, 0], 0)
#
#
# def test_aggregate_metric():
#     # output[img_idx]['pairwise_metric'][idx1][idx2]
#     N = 4
#     img_idx = 20
#     metric_dict = {img_idx: {'pairwise_metric': {}}}
#     for idx1 in range(1, N + 1):
#         metric_dict[img_idx]['pairwise_metric'][idx1 - 1] = {}
#         for idx2 in range(1, N + 1):
#             metric_dict[img_idx]['pairwise_metric'][idx1 - 1][idx2 - 1] = {'RMSE': idx2 + N * (idx1 - 1)}
#
#     output = aggregate_metric(metric_dict)
#     N2 = N * N
#     assert equal(output[img_idx]['RMSE'], ((N2 + 1)) / 2)
