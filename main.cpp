#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>
constexpr float EPSILON = 1e-4f;
constexpr float PI = 3.14159265358979323846f;
struct Vec3 {
  float x;
  float y;
  float z;
  Vec3(float x, float y, float z) : x(x), y(y), z(z) {}
  Vec3() : x(0), y(0), z(0) {}
  Vec3 operator+(const Vec3 &other) const {
    return Vec3(x + other.x, y + other.y, z + other.z);
  }
  Vec3 operator-(const Vec3 &other) const {
    return Vec3(x - other.x, y - other.y, z - other.z);
  }
  Vec3 operator*(float other) const {
    return Vec3(x * other, y * other, z * other);
  }
  Vec3 operator*(Vec3 other) {
    return Vec3(x * other.x, y * other.y, z * other.z);
  }
  float length() const { return sqrt(x * x + y * y + z * z); }
};
inline Vec3 normalize(Vec3 v) { return v * (1.0f / v.length()); }
inline float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline float clamp(float x, float min, float max) {
  return x < min ? min : (x > max ? max : x);
}
float linear_to_srgb(float x) {
  x = clamp(x, 0.0f, 1.0f);
  if (x <= 0.0031308f)
    return 12.92f * x;
  return 1.055f * std::pow(x, 1.0f / 2.4f) - 0.055f;
}
struct Ray {
  Vec3 origin;
  Vec3 direction;
  Ray(Vec3 origin, Vec3 direction) : origin(origin), direction(direction) {}
  Vec3 at(float t) const { return origin + direction * t; }
};
struct Sphere {
  Sphere(float radius, Vec3 center) : radius(radius), center(center) {}
  float radius;
  Vec3 center;
  Vec3 color;
  bool is_light = false;
};
struct HitInfo {
  bool is_hit;
  float t;
  Vec3 position;
  Vec3 normal;
  Vec3 color;
  bool is_light;
};
HitInfo intersect(const Ray &ray, const Sphere &sphere) {
  Vec3 oc = ray.origin - sphere.center;
  float b = 2.0f * dot(oc, ray.direction);
  float c = dot(oc, oc) - sphere.radius * sphere.radius;
  float discriminant = b * b - 4.0f * c;
  if (discriminant < 0.0f)
    return {false, 0.0f, Vec3(), Vec3(), Vec3(), false};
  float sqrt_d = sqrt(discriminant);
  float t0 = (-b - sqrt_d) * 0.5f;
  float t1 = (-b + sqrt_d) * 0.5f;
  float t = t0;
  if (t < EPSILON)
    t = t1;
  if (t < EPSILON)
    return {false, 0.0f, Vec3(), Vec3(), Vec3(), false};
  Vec3 position = ray.at(t);
  Vec3 normal = normalize(position - sphere.center);
  return {true, t, position, normal, sphere.color, sphere.is_light};
}
float random_float() {
  thread_local std::mt19937 gen(std::minstd_rand{}());
  thread_local std::uniform_real_distribution<float> dis(0.0f, 1.0f);
  return dis(gen);
}
Vec3 random_in_hemisphere(const Vec3 &normal) {
  float z = random_float();
  float r = sqrt(1.0f - z * z);
  float phi = 2.0f * PI * random_float();
  Vec3 dir(r * std::cos(phi), z, r * std::sin(phi));
  if (dot(dir, normal) < 0.0f)
    dir = dir * -1.0f;
  return dir;
}
Vec3 f_r(Vec3 wi, Vec3 wo, Vec3 color) { return color * (1.0f / PI); }
float pdf(Vec3 wi) { return 1.0f / (2.0f * PI); }
Vec3 Lo(Ray ray, const std::vector<Sphere> &spheres, int depth) {
  if (depth <= 0)
    return Vec3();
  float rr_prob = 0.9f;
  if (random_float() > rr_prob)
    return Vec3();
  HitInfo nearest_hit;
  nearest_hit.t = 1e10;
  nearest_hit.is_hit = false;
  for (auto &sphere : spheres) {
    HitInfo hit = intersect(ray, sphere);
    if (hit.is_hit && hit.t < nearest_hit.t)
      nearest_hit = hit;
  }
  if (!nearest_hit.is_hit)
    return Vec3();
  if (nearest_hit.is_light) {
    Vec3 Le = nearest_hit.color * 30.0f;
    return Le;
  }
  Vec3 wo = normalize(ray.direction);
  Vec3 wi = random_in_hemisphere(nearest_hit.normal);
  float pdf_val = pdf(wi);
  Vec3 f_r_val = f_r(wi, wo, nearest_hit.color);
  float cos_theta = dot(wi, nearest_hit.normal);
  Ray new_ray = Ray(nearest_hit.position + nearest_hit.normal * EPSILON, wi);
  Vec3 Li = Lo(new_ray, spheres, depth - 1);
  return f_r_val * Li * cos_theta * (1.0f / (rr_prob * pdf_val));
}
int main() {
  int image_width = 1080, image_height = 1080;
  std::vector<Vec3> framebuffer(image_width * image_height);
  int samples_per_pixel = 20000;
  constexpr float R = 1000.0f;
  Sphere left_wall = Sphere(R, Vec3(-1003.0f, 0.0f, -6.0f));
  left_wall.color = Vec3(0.75f, 0.12f, 0.10f);
  Sphere right_wall = Sphere(R, Vec3(1003.0f, 0.0f, -6.0f));
  right_wall.color = Vec3(0.12f, 0.45f, 0.12f);
  Sphere back_wall = Sphere(R, Vec3(0.0f, 0.0f, -1006.0f));
  back_wall.color = Vec3(1.0f, 1.0f, 1.0f);
  Sphere floor = Sphere(R, Vec3(0.0f, -1002.6f, -6.0f));
  floor.color = Vec3(1.0f, 1.0f, 1.0f);
  Sphere ceiling = Sphere(R, Vec3(0.0f, 1002.6f, -6.0f));
  ceiling.color = Vec3(1.0f, 1.0f, 1.0f);
  Sphere light = Sphere(0.35f, Vec3(0.0f, 2.6f, -4.0f));
  light.color = Vec3(1.0f, 1.0f, 1.0f);
  light.is_light = true;
  Sphere big_sphere = Sphere(1.3f, Vec3(-1.0f, -1.6f, -5.0f));
  big_sphere.color = Vec3(1.0f, 1.0f, 1.0f);
  Sphere small_sphere = Sphere(0.8f, Vec3(1.3f, -2.0f, -4.0f));
  small_sphere.color = Vec3(1.0f, 1.0f, 1.0f);
  std::vector<Sphere> spheres = {left_wall,  right_wall,  back_wall,
                                 floor,      ceiling,     light,
                                 big_sphere, small_sphere};
  int done = 0;
#pragma omp parallel for schedule(dynamic)
  for (int j = 0; j < image_height; j++) {
    #pragma omp critical
    std::cout << '\r' << ++done * 100 / image_height << '%' << std::flush;
    for (int i = 0; i < image_width; i++) {
      Vec3 color;
      for (int s = 0; s < samples_per_pixel; s++) {
        float u = (double(i) + random_float()) / image_width;
        float v = (double(j) + random_float()) / image_height;
        u = (u - 0.5) * 2;
        v = (v - 0.5) * -2;
        Vec3 ray_direction = normalize(Vec3(u, v, -1));
        Ray ray = Ray(Vec3(0.0f, 0.0f, 0.0f), ray_direction);
        color = color + Lo(ray, spheres, 10);
      }
      color = color * (1.0f / samples_per_pixel);
      framebuffer[j * image_width + i] = color;
    }
  }
  std::ofstream file("output.ppm");
  file << "P3\n" << image_width << ' ' << image_height << '\n' << "255\n";
  for (int i = 0; i < image_width * image_height; ++i) {
    Vec3 color = framebuffer[i];
    int ir = static_cast<int>(linear_to_srgb(color.x) * 255.0f);
    int ig = static_cast<int>(linear_to_srgb(color.y) * 255.0f);
    int ib = static_cast<int>(linear_to_srgb(color.z) * 255.0f);
    file << ir << ' ' << ig << ' ' << ib << '\n';
  }
}