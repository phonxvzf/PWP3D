#include <cstdint>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "main.hpp"
#include "globals.hpp"
#include "shader.hpp"
#include "glad.h"
#include "model.hpp"
#include "fitter.hpp"

#define ROT_STEP_SIZE 0.01f
#define TRANS_STEP_SIZE 0.01f

void add_alpha_channel(const cv::Mat& mat, cv::Mat& dst) {
  std::vector<cv::Mat> mat_channels;
  cv::split(mat, mat_channels);
  cv::Mat alpha(mat.rows, mat.cols, CV_8UC1, cv::Scalar(255));
  mat_channels.push_back(alpha);
  cv::merge(mat_channels, dst);
}

inline float convert_angle(float step, const float step_size) {
  return (-360.0f + step * step_size);
}

inline float convert_trans(float step, const float step_size) {
  return (-100.0f + step * step_size);
}

GLuint frame_texture;
GLuint frame_vao, frame_vbo, frame_ebo;
shader *frame_shader, *model_shader;
model *object_model;
void init_gl(const int tx_width, const int tx_height) {
  // Initialize frame VAO
  glGenVertexArrays(1, &frame_vao);
  glBindVertexArray(frame_vao);

  // Initialize frame VBO
  glGenBuffers(1, &frame_vbo);
  glBindBuffer(GL_ARRAY_BUFFER, frame_vbo);
  const GLfloat vertices[] = {
    1.0f, 1.0f, 0.0f, 1.0f, 1.0f,
    1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
    -1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
    -1.0f, 1.0f, 0.0f, 0.0f, 1.0f
  };
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STREAM_READ | GL_STREAM_DRAW);

  // Initialize frame EBO
  glGenBuffers(1, &frame_ebo);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, frame_ebo);
  const GLuint order[] = {
    0, 1, 2, 3, 0, 2
  };
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(order), order, GL_STREAM_READ | GL_STREAM_DRAW);
  
  // Initialize shader program
  frame_shader = new shader("shader/frame.vert", "shader/frame.frag");
  model_shader = new shader("shader/model.vert", "shader/model.frag");
  
  // Enable vertex attributes
  // [v][v][v][t][t]
  frame_shader->use();
  GLint vert_position_loc = glGetAttribLocation(frame_shader->program(), "in_position");
  GLint tex_coords_loc = glGetAttribLocation(frame_shader->program(), "in_tex_coords");
  const size_t stride = 5 * sizeof(GLfloat);
  glVertexAttribPointer(vert_position_loc, 3, GL_FLOAT, GL_FALSE, stride, 0);
  glVertexAttribPointer(tex_coords_loc, 2, GL_FLOAT, GL_FALSE, stride, 0);
  glEnableVertexAttribArray(vert_position_loc);
  glEnableVertexAttribArray(tex_coords_loc);

  // Unbind stuff
  glBindVertexArray(0);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

  // Initialize frame texture
  glGenTextures(1, &frame_texture);
  glBindTexture(GL_TEXTURE_2D, frame_texture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexImage2D(
      GL_TEXTURE_2D,
      0,
      GL_RGBA,
      tx_width,
      tx_height,
      0,
      GL_BGRA,
      GL_UNSIGNED_INT_8_8_8_8_REV,
      nullptr
      );
  glBindTexture(GL_TEXTURE_2D, 0);

  // Initialize deer model
  // deer_model = new model("long.obj", *model_shader);
  object_model = new model("long.obj", *model_shader);
}

void clean_gl() {
  object_model->clean();
  frame_shader->clean();
  model_shader->clean();
  glDeleteTextures(1, &frame_texture);
  glDeleteBuffers(1, &frame_vao);
  glDeleteBuffers(1, &frame_vbo);
  glDeleteBuffers(1, &frame_ebo);
  delete frame_shader;
}

struct draw_data {
  cv::Mat *frame;
  cv::Mat *extrinsic_rot;

  float *extrinsic_trans;
  float camera_fx, camera_fy;
  float camera_cx, camera_cy;

  bool ready = false;

  // controls
  float rot_x = 0;
  float rot_y = 0;
  float rot_z = 0;

  float trans_x = 0;
  float trans_y = 0;
  float trans_z = 0;

} draw_data_ctrl;

void draw_gl(void *params) {
  struct draw_data data = *static_cast<struct draw_data*>(params);

  cv::Mat frame = *data.frame;
  cv::Mat flipped_frame;
  cv::flip(frame, flipped_frame, 0); // OpenGL will flip the texture vertically
  cv::Mat frame_walpha(flipped_frame.cols, flipped_frame.rows, CV_8UC4);
  add_alpha_channel(flipped_frame, frame_walpha);

  // feed frame texture to GPU
  glBindTexture(GL_TEXTURE_2D, frame_texture);
  glTexSubImage2D(
      GL_TEXTURE_2D,
      0,
      0,
      0,
      frame.cols,
      frame.rows,
      GL_BGRA,
      GL_UNSIGNED_INT_8_8_8_8_REV,
      frame_walpha.data
  );

  glClearColor(1.0f, 0.0f, 0.0f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // draw frame
  glBindVertexArray(frame_vao);
  frame_shader->use();
  const GLint u_tex_sampler_loc = glGetUniformLocation(frame_shader->program(), "u_tex_sampler");
  glUniform1i(u_tex_sampler_loc, 0);
  glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
  frame_shader->detach();
  glBindVertexArray(0);
  glBindTexture(GL_TEXTURE_2D, 0);

  // draw object
  // enable wireframe mode
  glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

  model_shader->use();
  const GLint u_model_loc = glGetUniformLocation(model_shader->program(), "u_model");
  const GLint u_view_loc = glGetUniformLocation(model_shader->program(), "u_view");
  const GLint u_intrinsic_loc = glGetUniformLocation(model_shader->program(), "u_intrinsic");
  const GLint u_extrinsic_rot_loc = glGetUniformLocation(model_shader->program(), "u_extrinsic_rot");
  const GLint u_extrinsic_trans_loc = glGetUniformLocation(model_shader->program(), "u_extrinsic_trans");

  glm::mat4 view_mat(1.0f);
  glm::mat4 model_mat(1.0f);
  glm::mat3 intrinsic_mat(0.0f);

  intrinsic_mat[0][0] = draw_data_ctrl.camera_fx;
  intrinsic_mat[1][1] = draw_data_ctrl.camera_fy;
  intrinsic_mat[0][2] = draw_data_ctrl.camera_cx;
  intrinsic_mat[1][2] = draw_data_ctrl.camera_cy;
  intrinsic_mat[2][2] = 1.0f;

  const float rx = glm::radians(draw_data_ctrl.rot_x);
  const float ry = glm::radians(draw_data_ctrl.rot_y);
  const float rz = glm::radians(draw_data_ctrl.rot_z);

  const float tx = draw_data_ctrl.trans_x;
  const float ty = draw_data_ctrl.trans_y;
  const float tz = draw_data_ctrl.trans_z;

  glm::mat3 extrinsic_rot_mat =
      glm::rotate(glm::mat4(1.0f), rx, glm::vec3(1.0f, 0.0f, 0.0f))
    * glm::rotate(glm::mat4(1.0f), ry, glm::vec3(0.0f, 1.0f, 0.0f)) 
    * glm::rotate(glm::mat4(1.0f), rz, glm::vec3(0.0f, 0.0f, 1.0f));

  glm::vec3 extrinsic_trans_vec(tx, ty, tz);

  glUniformMatrix4fv(u_model_loc, 1, GL_FALSE, glm::value_ptr(model_mat));
  glUniformMatrix4fv(u_view_loc, 1, GL_FALSE, glm::value_ptr(view_mat));
  glUniformMatrix3fv(u_extrinsic_rot_loc, 1, GL_TRUE, glm::value_ptr(extrinsic_rot_mat));
  glUniformMatrix3fv(u_intrinsic_loc, 1, GL_TRUE, glm::value_ptr(intrinsic_mat));
  glUniform3fv(u_extrinsic_trans_loc, 1, glm::value_ptr(extrinsic_trans_vec));

  object_model->draw();
  model_shader->detach();

  // disable wireframe mode
  glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

void read_calib(const std::string& file_path) {
  FILE* f = std::fopen(file_path.c_str(), "r");
  if (f == nullptr) {
    std::cerr << "[ERROR] Could not open camera calibration file" << std::endl;
    std::exit(1);
  }

  int w, h;
  fscanf(f, "Perseus_CalFile\n%d %d\n", &w, &h);
  fscanf(f, "%f%f%f%f",
      &draw_data_ctrl.camera_fx, &draw_data_ctrl.camera_fy,
      &draw_data_ctrl.camera_cx, &draw_data_ctrl.camera_cy);
  std::fclose(f);
}

int main(int argc, char** argv) {
  if (argc != 7) {
    std::cerr << "usage: PWP3DAPP [obj] [src] [calib] [mask] [hist] [hist_mask]" << std::endl;
    return EXIT_FAILURE;
  }

  std::string obj_path(argv[1]);
  std::string src_path(argv[2]);
  std::string calib_path(argv[3]);
  std::string mask_path(argv[4]);
  std::string hist_path(argv[5]);
  std::string hist_mask_path(argv[6]);

  read_calib(calib_path);

  cv::namedWindow(MONITOR_TITLE, cv::WINDOW_KEEPRATIO | cv::WINDOW_OPENGL);
  cv::namedWindow(CTRL_TITLE, cv::WINDOW_GUI_EXPANDED);

  cv::resizeWindow(MONITOR_TITLE, cv::Size(640, 480));
  cv::resizeWindow(CTRL_TITLE, cv::Size(640, 480));
  cv::setOpenGlContext(MONITOR_TITLE);

  if (gladLoadGL()) {
    std::cout 
      << "OpenGL Loaded Successfully\n"
      << "=========================="
      << std::endl;
    std::cout << "Vendor\t\t: " << glGetString(GL_VENDOR) << std::endl;
    std::cout << "Renderer\t: " << glGetString(GL_RENDERER) << std::endl;
    std::cout << "GL Version\t: " << glGetString(GL_VERSION) << std::endl;
    std::cout << "GLSL Version\t: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;
  } else {
    std::cerr << "Unable to load GL" << std::endl;
    return 1;
  }

  cv::Mat camera_matrix(3, 3, CV_64FC1, (double*) CAM_MATRIX_VALUES);

  cv::resizeWindow(MONITOR_TITLE, 640, 480);
  init_gl(640, 480);

  std::vector<double> rvec, tvec;

  draw_data_ctrl.ready = true;
  cv::Mat frame = cv::imread(src_path);
  draw_data_ctrl.frame = &frame;
  cv::Mat rvec_mat;

  // Create pose estimator
  fitter f;
  f.set_size(640, 480)
    .set_obj(obj_path)
    .set_view(calib_path)
    .set_image(cv::imread(src_path))
    .set_mask(cv::imread(mask_path))
    .set_hist(cv::imread(hist_path))
    .set_hist_mask(hist_mask_path)
    .set_step_size(0.2f, 0.5f, 0.5f, 10.0f)
    .set_initial_pose(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 0.0f));

  const int max_rot_step = 720 / ROT_STEP_SIZE;
  const int max_trans_step = 200 / TRANS_STEP_SIZE;
  int ts_x, ts_y, ts_z, rs_x, rs_y, rs_z;
  ts_x = max_trans_step >> 1;
  ts_y = max_trans_step >> 1;
  ts_z = max_trans_step >> 1;
  rs_x = max_rot_step >> 1;
  rs_y = max_rot_step >> 1;
  rs_z = max_rot_step >> 1;
  cv::createTrackbar("tX", CTRL_TITLE, &ts_x, max_trans_step);
  cv::createTrackbar("tY", CTRL_TITLE, &ts_y, max_trans_step);
  cv::createTrackbar("tZ", CTRL_TITLE, &ts_z, max_trans_step);
  cv::createTrackbar("rX", CTRL_TITLE, &rs_x, max_rot_step);
  cv::createTrackbar("rY", CTRL_TITLE, &rs_y, max_rot_step);
  cv::createTrackbar("rZ", CTRL_TITLE, &rs_z, max_rot_step);

  cv::setOpenGlDrawCallback(MONITOR_TITLE, draw_gl, &draw_data_ctrl);
  bool state_fitting = false;
  bool continue_loop = true;
  char pose_status[256];
  while (continue_loop) {
    draw_data_ctrl.rot_x = convert_angle(rs_x, ROT_STEP_SIZE);
    draw_data_ctrl.rot_y = convert_angle(rs_y, ROT_STEP_SIZE);
    draw_data_ctrl.rot_z = convert_angle(rs_z, ROT_STEP_SIZE);

    draw_data_ctrl.trans_x = convert_trans(ts_x, TRANS_STEP_SIZE);
    draw_data_ctrl.trans_y = convert_trans(ts_y, TRANS_STEP_SIZE);
    draw_data_ctrl.trans_z = convert_trans(ts_z, TRANS_STEP_SIZE);
    
    // update pose
    if (state_fitting) f.fit();
    std::sprintf(
        pose_status,
        "[%s] T: < %f, %f, %f > R: %f, %f, %f\n",
        state_fitting ? "fitting..." : "idle",
        draw_data_ctrl.trans_x, draw_data_ctrl.trans_y, draw_data_ctrl.trans_z,
        draw_data_ctrl.rot_x, draw_data_ctrl.rot_y, draw_data_ctrl.rot_z);
    cv::displayStatusBar(CTRL_TITLE, cv::String(pose_status));
    cv::updateWindow(MONITOR_TITLE);

    // check for keyboard input
    char key = cv::waitKey(25);
    switch (key) {
      case 'q':
        if (state_fitting) {
          f.shutdown();
        }
        continue_loop = false;
        break;
      case ' ':
        state_fitting = !state_fitting;
        if (state_fitting) {
          f.set_initial_pose(
              glm::vec3(draw_data_ctrl.trans_x, draw_data_ctrl.trans_y, draw_data_ctrl.trans_z),
              glm::vec3(draw_data_ctrl.rot_x, draw_data_ctrl.rot_y, draw_data_ctrl.rot_z)
              );
          f.init();
        }
        break;
      default:
        break;
    }
  }

  cv::setOpenGlDrawCallback(MONITOR_TITLE, nullptr);
  cv::destroyWindow(MONITOR_TITLE);
  clean_gl();
  return 0;
}
