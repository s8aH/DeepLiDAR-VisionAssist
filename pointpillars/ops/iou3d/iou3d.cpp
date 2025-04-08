// Modified file: pointpillars/ops/iou3d/iou3d.cpp

// If you have CUDA, define WITH_CUDA (e.g., via a compiler flag -DWITH_CUDA)
// Otherwise, the following stub implementations will be used.

#ifdef WITH_CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include <cstdint>
#include <vector>

#define CHECK_CUDA(x) \
  TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))
#define THREADS_PER_BLOCK_NMS (sizeof(unsigned long long) * 8)

#define CHECK_ERROR(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

// Forward declarations for CUDA launcher functions.
void boxesoverlapLauncher(const int num_a, const float *boxes_a,
                          const int num_b, const float *boxes_b,
                          float *ans_overlap);
void boxesioubevLauncher(const int num_a, const float *boxes_a, const int num_b,
                         const float *boxes_b, float *ans_iou);
void nmsLauncher(const float *boxes, unsigned long long *mask, int boxes_num,
                 float nms_overlap_thresh);
void nmsNormalLauncher(const float *boxes, unsigned long long *mask,
                       int boxes_num, float nms_overlap_thresh);

int boxes_overlap_bev_gpu(at::Tensor boxes_a, at::Tensor boxes_b,
                          at::Tensor ans_overlap) {
  // boxes_a: (N, 5) [x1, y1, x2, y2, ry]
  // boxes_b: (M, 5)
  // ans_overlap: (N, M)
  CHECK_INPUT(boxes_a);
  CHECK_INPUT(boxes_b);
  CHECK_INPUT(ans_overlap);

  int num_a = boxes_a.size(0);
  int num_b = boxes_b.size(0);

  const float *boxes_a_data = boxes_a.data_ptr<float>();
  const float *boxes_b_data = boxes_b.data_ptr<float>();
  float *ans_overlap_data = ans_overlap.data_ptr<float>();

  boxesoverlapLauncher(num_a, boxes_a_data, num_b, boxes_b_data, ans_overlap_data);

  return 1;
}

int boxes_iou_bev_gpu(at::Tensor boxes_a, at::Tensor boxes_b,
                      at::Tensor ans_iou) {
  // boxes_a: (N, 5) [x1, y1, x2, y2, ry]
  // boxes_b: (M, 5)
  // ans_iou: (N, M)
  CHECK_INPUT(boxes_a);
  CHECK_INPUT(boxes_b);
  CHECK_INPUT(ans_iou);

  int num_a = boxes_a.size(0);
  int num_b = boxes_b.size(0);

  const float *boxes_a_data = boxes_a.data_ptr<float>();
  const float *boxes_b_data = boxes_b.data_ptr<float>();
  float *ans_iou_data = ans_iou.data_ptr<float>();

  boxesioubevLauncher(num_a, boxes_a_data, num_b, boxes_b_data, ans_iou_data);

  return 1;
}

int nms_gpu(at::Tensor boxes, at::Tensor keep,
            float nms_overlap_thresh, int device_id) {
  // boxes: (N, 5) [x1, y1, x2, y2, ry]
  // keep: (N)
  CHECK_INPUT(boxes);
  CHECK_CONTIGUOUS(keep);
  cudaSetDevice(device_id);

  int boxes_num = boxes.size(0);
  const float *boxes_data = boxes.data_ptr<float>();
  int64_t *keep_data = keep.data_ptr<int64_t>();

  const int col_blocks = DIVUP(boxes_num, THREADS_PER_BLOCK_NMS);
  unsigned long long *mask_data = nullptr;
  CHECK_ERROR(cudaMalloc((void **)&mask_data,
                         boxes_num * col_blocks * sizeof(unsigned long long)));
  nmsLauncher(boxes_data, mask_data, boxes_num, nms_overlap_thresh);

  std::vector<unsigned long long> mask_cpu(boxes_num * col_blocks);
  CHECK_ERROR(cudaMemcpy(&mask_cpu[0], mask_data,
                         boxes_num * col_blocks * sizeof(unsigned long long),
                         cudaMemcpyDeviceToHost));
  cudaFree(mask_data);

  unsigned long long *remv_cpu = new unsigned long long[col_blocks]();
  int num_to_keep = 0;
  for (int i = 0; i < boxes_num; i++) {
    int nblock = i / THREADS_PER_BLOCK_NMS;
    int inblock = i % THREADS_PER_BLOCK_NMS;
    if (!(remv_cpu[nblock] & (1ULL << inblock))) {
      keep_data[num_to_keep++] = i;
      unsigned long long *p = &mask_cpu[0] + i * col_blocks;
      for (int j = nblock; j < col_blocks; j++) {
        remv_cpu[j] |= p[j];
      }
    }
  }
  delete[] remv_cpu;
  if (cudaSuccess != cudaGetLastError()) printf("Error!\n");

  return num_to_keep;
}

int nms_normal_gpu(at::Tensor boxes, at::Tensor keep,
                   float nms_overlap_thresh, int device_id) {
  // boxes: (N, 5) [x1, y1, x2, y2, ry]
  // keep: (N)
  CHECK_INPUT(boxes);
  CHECK_CONTIGUOUS(keep);
  cudaSetDevice(device_id);

  int boxes_num = boxes.size(0);
  const float *boxes_data = boxes.data_ptr<float>();
  int64_t *keep_data = keep.data_ptr<int64_t>();

  const int col_blocks = DIVUP(boxes_num, THREADS_PER_BLOCK_NMS);
  unsigned long long *mask_data = nullptr;
  CHECK_ERROR(cudaMalloc((void **)&mask_data,
                         boxes_num * col_blocks * sizeof(unsigned long long)));
  nmsNormalLauncher(boxes_data, mask_data, boxes_num, nms_overlap_thresh);

  std::vector<unsigned long long> mask_cpu(boxes_num * col_blocks);
  CHECK_ERROR(cudaMemcpy(&mask_cpu[0], mask_data,
                         boxes_num * col_blocks * sizeof(unsigned long long),
                         cudaMemcpyDeviceToHost));
  cudaFree(mask_data);

  unsigned long long *remv_cpu = new unsigned long long[col_blocks]();
  int num_to_keep = 0;
  for (int i = 0; i < boxes_num; i++) {
    int nblock = i / THREADS_PER_BLOCK_NMS;
    int inblock = i % THREADS_PER_BLOCK_NMS;
    if (!(remv_cpu[nblock] & (1ULL << inblock))) {
      keep_data[num_to_keep++] = i;
      unsigned long long *p = &mask_cpu[0] + i * col_blocks;
      for (int j = nblock; j < col_blocks; j++) {
        remv_cpu[j] |= p[j];
      }
    }
  }
  delete[] remv_cpu;
  if (cudaSuccess != cudaGetLastError()) printf("Error!\n");

  return num_to_keep;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("boxes_overlap_bev_gpu", &boxes_overlap_bev_gpu, "oriented boxes overlap");
  m.def("boxes_iou_bev_gpu", &boxes_iou_bev_gpu, "oriented boxes iou");
  m.def("nms_gpu", &nms_gpu, "oriented nms gpu");
  m.def("nms_normal_gpu", &nms_normal_gpu, "nms gpu");
}

#else  // CPU-only stubs

#include <torch/extension.h>
#include <stdexcept>

int boxes_overlap_bev_gpu(at::Tensor boxes_a, at::Tensor boxes_b, at::Tensor ans_overlap) {
  throw std::runtime_error("boxes_overlap_bev_gpu is not supported in CPU-only mode");
}

int boxes_iou_bev_gpu(at::Tensor boxes_a, at::Tensor boxes_b, at::Tensor ans_iou) {
  throw std::runtime_error("boxes_iou_bev_gpu is not supported in CPU-only mode");
}

int nms_gpu(at::Tensor boxes, at::Tensor keep, float nms_overlap_thresh, int device_id) {
  throw std::runtime_error("nms_gpu is not supported in CPU-only mode");
}

int nms_normal_gpu(at::Tensor boxes, at::Tensor keep, float nms_overlap_thresh, int device_id) {
  throw std::runtime_error("nms_normal_gpu is not supported in CPU-only mode");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("boxes_overlap_bev_gpu", &boxes_overlap_bev_gpu, "oriented boxes overlap (CPU stub)");
  m.def("boxes_iou_bev_gpu", &boxes_iou_bev_gpu, "oriented boxes iou (CPU stub)");
  m.def("nms_gpu", &nms_gpu, "oriented nms gpu (CPU stub)");
  m.def("nms_normal_gpu", &nms_normal_gpu, "nms gpu (CPU stub)");
}

#endif  // WITH_CUDA
