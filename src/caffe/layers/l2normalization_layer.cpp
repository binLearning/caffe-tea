#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/l2normalization_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void L2NormalizationLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(),
  //    bottom[0]->height(), bottom[0]->width());
  //squared_.Reshape(bottom[0]->num(), bottom[0]->channels(), 
  //  bottom[0]->height(), bottom[0]->width());
  
  top[0]->Reshape(bottom[0]->shape());
  squared_.Reshape(bottom[0]->shape());
}

template <typename Dtype>
void L2NormalizationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* squared_data = squared_.mutable_cpu_data();
  
  // NCHW
  int num = bottom[0]->num(); // batch size
  int dim = bottom[0]->count() / num;
  
  caffe_sqr<Dtype>(bottom[0]->count(), bottom_data, squared_data);
  
  for (int n = 0; n < num; ++n) {
    int offset = n * dim;
    Dtype l2norm = pow(caffe_cpu_asum<Dtype>(dim, squared_data + offset), -0.5);
    caffe_cpu_scale<Dtype>(dim, l2norm, bottom_data + offset, top_data + offset);
  }
}

template <typename Dtype>
void L2NormalizationLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  
  int num = top[0]->num();
  int dim = top[0]->count() / num;
  
  for (int n = 0; n < num; ++n) {
    int offset = n * dim;
    Dtype dot_top_data_diff = caffe_cpu_dot(dim, top_data + offset, top_diff + offset);
    caffe_cpu_scale(dim, dot_top_data_diff, top_data + offset, bottom_diff + offset);
    caffe_sub(dim, top_diff + offset, bottom_diff + offset, bottom_diff + offset);
    Dtype bottom_data_norm = pow(caffe_cpu_dot(dim, bottom_data + offset, bottom_data + offset), -0.5);
    caffe_cpu_scale(dim, bottom_data_norm, bottom_diff + offset, bottom_diff + offset);
  }
}

#ifdef CPU_ONLY
STUB_GPU(L2NormalizationLayer);
#endif

INSTANTIATE_CLASS(L2NormalizationLayer);
REGISTER_LAYER_CLASS(L2Normalization);

}  // namespace caffe
