#include "fitter.hpp"

fitter& fitter::set_image(const cv::Mat& image) {
  this->m_image = image;
  return *this;
}

fitter& fitter::set_mask(const cv::Mat& mask) {
  this->m_mask = mask;
  ImageUtils::Instance()->LoadImageFromCVMat(this->m_view->videoMask, (cv::Mat&) mask);
  return *this;
}

fitter& fitter::set_hist(const cv::Mat& hist) {
  this->m_hist = hist;
  ImageUtils::Instance()->LoadImageFromCVMat(this->m_obj->histSources[0], (cv::Mat&) hist);
  return *this;
}

fitter& fitter::set_hist_mask(const std::string& file_path) {
  ImageUtils::Instance()->LoadImageFromFile(this->m_obj->histMasks[0], (char*) file_path.c_str(), 1);
  return *this;
}

fitter& fitter::set_size(int width, int height) {
  this->m_width = width;
  this->m_height = height;
  return *this;
}

fitter& fitter::set_step_size(float r, float tx, float ty, float tz) {
  this->m_step_size = new StepSize3D(r, tx, ty, tz);
  this->m_obj->stepSize[0] = this->m_step_size;
  return *this;
}

fitter& fitter::set_initial_pose(const glm::vec3& init_trans, const glm::vec3& init_rot) {
  this->m_init_trans = init_trans;
  this->m_init_rot = init_rot;
  this->m_obj->initialPose[0]->SetFrom(
      init_trans.x,
      init_trans.y,
      init_trans.z,
      init_rot.x,
      init_rot.y,
      init_rot.z
      );
  return *this;
}
    
fitter& fitter::set_obj(const std::string& file_path) {
  this->m_obj = new Object3D(0, 1, (char*) file_path.c_str(), this->m_width, this->m_height);
  return *this;
}

fitter& fitter::set_view(const std::string& calib_file_path) {
  this->m_view = new View3D(0, (char*) calib_file_path.c_str(), this->m_width, this->m_height);
  return *this;
}

void fitter::get_pose(glm::vec3& out_tvec, glm::vec3& out_rvec) const {
  out_tvec = this->m_pose_trans;
  out_rvec = this->m_pose_rot;
}

void fitter::init() {
  HistogramEngine::Instance()->UpdateVarBinHistogram(
      this->m_obj, this->m_view, this->m_obj->histSources[0],
      this->m_obj->histMasks[0], this->m_view->videoMask
      );

  this->m_iter_conf.width = this->m_width;
  this->m_iter_conf.height = this->m_height;
  this->m_iter_conf.iterViewIds[0] = 0;
  this->m_iter_conf.iterObjectCount[0] = 1;
  this->m_iter_conf.levelSetBandSize = 8;
  this->m_iter_conf.iterObjectIds[0][0] = 0;
  this->m_iter_conf.iterViewCount = 1;
  this->m_iter_conf.iterCount = 1;

  OptimisationEngine::Instance()->Initialise(this->m_width, this->m_height);

  ImageUChar4* tmp_image = new ImageUChar4(this->m_width, this->m_height);
  ImageUtils::Instance()->LoadImageFromCVMat(tmp_image, this->m_image);
  OptimisationEngine::Instance()->RegisterViewImage(this->m_view, tmp_image);
}

void fitter::fit() {
  for (int i = 0; i < 4; ++i) {
    switch (i % 4) {
      case 0:
        this->m_iter_conf.useCUDAEF = true;
        this->m_iter_conf.useCUDARender = true;
        break;
      case 1:
        this->m_iter_conf.useCUDAEF = false;
        this->m_iter_conf.useCUDARender = true;
        break;
      case 2:
        this->m_iter_conf.useCUDAEF = true;
        this->m_iter_conf.useCUDARender = false;
        break;
      case 3:
        this->m_iter_conf.useCUDAEF = false;
        this->m_iter_conf.useCUDARender = false;
        break;
    }
    OptimisationEngine::Instance()->Minimise(&this->m_obj, &this->m_view, &this->m_iter_conf);
  }

  // save pose estimation
  this->m_pose_trans = glm::vec3(
      this->m_obj->pose[0]->translation->x,
      this->m_obj->pose[0]->translation->y,
      this->m_obj->pose[0]->translation->z
      );

  Vector3D<float> euler_rot;
  this->m_obj->pose[0]->rotation->ToEuler(&euler_rot);
  this->m_pose_rot = glm::vec3(euler_rot.x, euler_rot.y, euler_rot.z);
}

void fitter::shutdown() {
  OptimisationEngine::Instance()->Shutdown();
  delete m_obj;
  delete m_view;
}
