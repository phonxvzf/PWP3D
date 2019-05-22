#ifndef FITTER_HPP
#define FITTER_HPP

#include <opencv2/opencv.hpp>
#include <glm/glm.hpp>

#include "../../PerseusLib/PerseusLib.h"

class fitter {
  private:
    cv::Mat m_image;
    cv::Mat m_mask;
    cv::Mat m_hist;

    Object3D* m_obj;
    View3D* m_view;
    StepSize3D* m_step_size;

    int m_width;
    int m_height;

    glm::vec3 m_init_trans;
    glm::vec3 m_init_rot;
    glm::vec3 m_pose_trans;
    glm::vec3 m_pose_rot;

    IterationConfiguration m_iter_conf;

  public:
    fitter() {};
    fitter(const fitter&) {};

    fitter& set_image(const cv::Mat& image);
    fitter& set_mask(const cv::Mat& mask);
    fitter& set_hist(const cv::Mat& hist);
    fitter& set_hist_mask(const std::string& file_path);
    fitter& set_size(int width, int height);
    fitter& set_step_size(float r, float tx, float ty, float tz);
    fitter& set_initial_pose(const glm::vec3& init_trans, const glm::vec3& init_rot);
    fitter& set_obj(const std::string& file_path);
    fitter& set_view(const std::string& calib_file_path);

    void get_pose(glm::vec3& out_tvec, glm::vec3& out_rvec) const;

    // flow: set values -> init() -> fit()* -> shutdown()
    void init();
    void fit();
    void shutdown();
};

#endif /* FITTER_HPP */
