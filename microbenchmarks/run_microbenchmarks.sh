python3 benchmark_encode_decode.py -n 2000 -q --arch cuda > log_encode_cuda.txt
python3 benchmark_encode_decode.py -n 2000 --arch cuda > log_encode_cuda_no_quant.txt
python3 benchmark_encode_decode.py -n 20 -q --arch x64 > log_encode_x64.txt
python3 benchmark_encode_decode.py -n 20 --arch x64 > log_encode_x64_no_quant.txt
python3 benchmark_bit_struct_store.py -n 2000 -q --arch cuda > log_store_cuda.txt
python3 benchmark_bit_struct_store.py -n 2000 -q --arch cuda --no-fusion > log_store_cuda_no_fusion.txt
python3 benchmark_bit_struct_store.py -n 2000 -q --arch cuda --no-ad > log_store_cuda_no_ad.txt
python3 benchmark_bit_struct_store.py -n 2000 -q --arch cuda --no-ad --no-fusion > log_store_cuda_off.txt
python3 benchmark_bit_struct_store.py -n 2000 -q --arch cuda > log_store_cuda_no_quant.txt
python3 benchmark_bit_struct_store.py -n 20 -q --arch x64 > log_store_x64.txt
python3 benchmark_bit_struct_store.py -n 20 -q --arch x64 --no-fusion > log_store_x64_no_fusion.txt
python3 benchmark_bit_struct_store.py -n 20 -q --arch x64 --no-ad > log_store_x64_no_ad.txt
python3 benchmark_bit_struct_store.py -n 20 -q --arch x64 --no-ad --no-fusion > log_store_x64_off.txt
python3 benchmark_bit_struct_store.py -n 20 --arch x64 > log_store_x64_no_quant.txt
python3 benchmark_partial_store.py -n 2000 -q --arch cuda > log_partial_cuda.txt
python3 benchmark_partial_store.py -n 2000 -q --arch cuda --no-fusion > log_partial_cuda_no_fusion.txt
python3 benchmark_partial_store.py -n 2000 -q --arch cuda --no-ad > log_partial_cuda_no_ad.txt
python3 benchmark_partial_store.py -n 2000 -q --arch cuda --no-ad --no-fusion > log_partial_cuda_off.txt
python3 benchmark_partial_store.py -n 2000 --arch cuda > log_partial_cuda_no_quant.txt
python3 benchmark_partial_store.py -n 20 -q --arch x64 > log_partial_x64.txt
python3 benchmark_partial_store.py -n 20 -q --arch x64 --no-fusion > log_partial_x64_no_fusion.txt
python3 benchmark_partial_store.py -n 20 -q --arch x64 --no-ad > log_partial_x64_no_ad.txt
python3 benchmark_partial_store.py -n 20 -q --arch x64 --no-ad --no-fusion > log_partial_x64_off.txt
python3 benchmark_partial_store.py -n 20 --arch x64 > log_partial_x64_no_quant.txt
python3 benchmark_matmul.py -n 1000 -q --arch cuda > log_matmul_cuda.txt
python3 benchmark_matmul.py -n 1000 -q --arch cuda --no-fusion > log_matmul_cuda_no_fusion.txt
python3 benchmark_matmul.py -n 1000 -q --arch cuda --no-ad > log_matmul_cuda_no_ad.txt
python3 benchmark_matmul.py -n 1000 -q --arch cuda --no-ad --no-fusion > log_matmul_cuda_off.txt
# python3 benchmark_matmul.py -n 1000 --arch cuda > log_matmul_cuda_no_quant.txt
python3 benchmark_matmul.py -n 20 -q --arch x64 > log_matmul_x64.txt
python3 benchmark_matmul.py -n 20 -q --arch x64 --no-fusion > log_matmul_x64_no_fusion.txt
python3 benchmark_matmul.py -n 20 -q --arch x64 --no-ad > log_matmul_x64_no_ad.txt
python3 benchmark_matmul.py -n 20 -q --arch x64 --no-ad --no-fusion > log_matmul_x64_off.txt
# python3 benchmark_matmul.py -n 20 --arch x64 > log_matmul_x64_no_quant.txt
