#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/l2normalization_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void L2NormalizationLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  Dtype* squared_data = squared_.mutable_gpu_data();
  
  // NCHW
  int num = bottom[0]->num(); // batch size
  int dim = bottom[0]->count() / num;
  
  caffe_gpu_powx(bottom[0]->count(), bottom_data, Dtype(2), squared_data);
  
  for (int n = 0; n < num; ++n) {
    int offset = num * dim;
    Dtype sum_squared_one_batch;
    caffe_gpu_asum<Dtype>(dim, squared_data + offset, &sum_squared_one_batch);
    caffe_gpu_scale<Dtype>(dim, Dtype(pow(sum_squared_one_batch, -0.5)), 
                           bottom_data + offset, top_data + offset);
  }
}

template <typename Dtype>
void L2NormalizationLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* top_data = top[0]->gpu_data();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  
  int num = top[0]->num();
  int dim = top[0]->count() / num;
  
  for (int n=0; n<num; ++n) {
    int offset = num * dim;
    Dtype dot_top_data_diff;
    Dtype bottom_data_norm;
    caffe_gpu_dot(dim, top_data + offset, top_diff + offset, &dot_top_data_diff);
    caffe_gpu_scale(dim, dot_top_data_diff, top_data + offset, bottom_diff + offset);
    caffe_gpu_sub(dim, top_diff + offset, bottom_diff + offset, bottom_diff + offset);
    caffe_gpu_dot(dim, bottom_data + offset, bottom_data + offset, &bottom_data_norm);
    caffe_gpu_scale(dim, Dtype(pow(bottom_data_norm, -0.5)), 
                    bottom_diff + offset, bottom_diff + offset);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(L2NormalizationLayer);

}  // namespace caffe