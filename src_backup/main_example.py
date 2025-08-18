
"""
예시 실행 스크립트:
  1) 학습:
     python v2x_training.py --v2aix_path V2AIX_Data/json/Mobile/V2X-only/Aachen \

                            --veremi_json_path VeReMi_Data/results/JSONlog-0-7-A0.json \

                            --veremi_ground_truth_path VeReMi_Data/results/GroundTruthJSONlog.json \

                            --out_dir artifacts --epochs 50
  2) 테스트:
     python v2x_testing.py --artifacts_dir artifacts \

                           --v2aix_path V2AIX_Data/json/Mobile/V2X-only/Aachen \

                           --veremi_json_path VeReMi_Data/results/JSONlog-0-7-A0.json \

                           --veremi_ground_truth_path VeReMi_Data/results/GroundTruthJSONlog.json
"""
print("See the docstring above for how to run training and testing from the command line.")
