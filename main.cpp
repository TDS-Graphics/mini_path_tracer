#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>
constexpr float PI = 3.14159265358979323846f;
constexpr float rr_prob = 0.9f;
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
  Vec3 operator-() const { return Vec3(-x, -y, -z); }
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
  Sphere const *sphere;
};
struct SampleLightInfo {
  Vec3 position;
  Vec3 normal;
  Vec3 Le;
  float pdf;
  Vec3 wk;
};
HitInfo intersect(const Ray &ray, const Sphere &sphere) {
  Vec3 oc = ray.origin - sphere.center;
  float b = 2.0f * dot(oc, ray.direction);
  float c = dot(oc, oc) - sphere.radius * sphere.radius;
  float discriminant = b * b - 4.0f * c;
  if (discriminant < 0.0f)
    return {false, 0.0f, Vec3(), Vec3(), &sphere};
  float sqrt_d = sqrt(discriminant);
  float t0 = (-b - sqrt_d) * 0.5f;
  float t1 = (-b + sqrt_d) * 0.5f;
  float t = t0;
  if (t < 1e-4f)
    t = t1;
  if (t < 1e-4f)
    return {false, 0.0f, Vec3(), Vec3(), &sphere};
  Vec3 position = ray.at(t);
  Vec3 normal = normalize(position - sphere.center);
  return {true, t, position, normal, &sphere};
}
float random_float() {
  thread_local std::mt19937 gen(std::random_device{}());
  thread_local std::uniform_real_distribution<float> dis(0.0f, 1.0f);
  return dis(gen);
}
Vec3 sample_bsdf(const Vec3 &normal) {
  float z = random_float();
  float r = std::sqrt(1.0f - z * z);
  float phi = 2.0f * PI * random_float();
  Vec3 local(r * std::cos(phi), r * std::sin(phi), z);
  Vec3 tangent;
  if (std::abs(normal.x) > 0.9f)
    tangent = normalize(Vec3(0.0f, 1.0f, 0.0f) - normal * normal.y);
  else
    tangent = normalize(Vec3(1.0f, 0.0f, 0.0f) - normal * normal.x);
  Vec3 bitangent = Vec3(normal.y * tangent.z - normal.z * tangent.y,
                        normal.z * tangent.x - normal.x * tangent.z,
                        normal.x * tangent.y - normal.y * tangent.x);
  return normalize(tangent * local.x + bitangent * local.y + normal * local.z);
}
float light_pdf(const Sphere &light, Vec3 p, Vec3 p_l) {
  Vec3 d = p_l - p;
  float dist2 = dot(d, d);
  Vec3 wi = normalize(d);
  Vec3 n = normalize(p_l - light.center);
  float cos_light = dot(n, -wi);
  if (cos_light <= 0.0f)
    return 0.0f;
  float pdf_area = 1.0f / (4.0f * PI * light.radius * light.radius);
  return pdf_area * dist2 / cos_light;
}
SampleLightInfo sample_light(const Sphere &light, Vec3 p) {
  float z = 1.0f - 2.0f * random_float();
  float r = sqrt(1.0f - z * z);
  float phi = 2.0f * PI * random_float();
  Vec3 n_l = Vec3(r * cos(phi), z, r * sin(phi));
  Vec3 p_l = light.center + n_l * light.radius;
  Vec3 d = p_l - p;
  float dist2 = dot(d, d);
  float dist = sqrt(dist2);
  Vec3 w_k = d * (1.0f / dist);
  float cos_light = dot(n_l, -w_k);
  if (cos_light <= 0.0f)
    return {p_l, n_l, Vec3(), 0.0f, w_k};
  float pdf_area = 1.0f / (4.0f * PI * light.radius * light.radius);
  float pdf = pdf_area * dist2 / cos_light;
  return {p_l, n_l, light.color * 30.0f, pdf, w_k};
}
bool visible(Vec3 p, Vec3 n, SampleLightInfo info,
             const std::vector<Sphere> &spheres) {
  Ray ray(p + n * 1e-4f, info.wk);
  for (const auto &sphere : spheres) {
    if (sphere.is_light)
      continue;
    HitInfo hit = intersect(ray, sphere);
    if (hit.is_hit && hit.t < (info.position - p).length() - 1e-4f)
      return false;
  }
  return true;
}
float mis_weight(float pdf_a, float pdf_b) {
  float a = pdf_a * pdf_a;
  float b = pdf_b * pdf_b;
  return a / (a + b);
}
Vec3 f_r(Vec3 wi, Vec3 wo, Vec3 color) { return color * (1.0f / PI); }
float bsdf_pdf(const Vec3 &wi) { return 1.0f / (2.0f * PI); }
Vec3 Ldir(Vec3 wo, Sphere const *light_sphereconst, HitInfo &hit_info,
          const std::vector<Sphere> &spheres) {
  SampleLightInfo info = sample_light(*light_sphereconst, hit_info.position);
  if (info.pdf <= 0.0f)
    return Vec3();
  if (!visible(hit_info.position, hit_info.normal, info, spheres))
    return Vec3();
  float cos_theta = dot(info.wk, hit_info.normal);
  Vec3 f_r_val = f_r(info.wk, wo, hit_info.sphere->color);
  float pdf_bsdf = bsdf_pdf(info.wk);
  float w = mis_weight(info.pdf, pdf_bsdf * rr_prob);
  Vec3 L_dir = f_r_val * info.Le * cos_theta * (w / info.pdf);
  return L_dir;
}
Vec3 Lo(Ray ray, const std::vector<Sphere> &spheres, int depth,
        float prev_pdf) {
  if (depth <= 0)
    return Vec3();
  HitInfo nearest_hit;
  nearest_hit.t = 1e10;
  nearest_hit.is_hit = false;
  Sphere const *light_sphere = nullptr;
  for (auto &sphere : spheres) {
    HitInfo hit = intersect(ray, sphere);
    if (hit.is_hit && hit.t < nearest_hit.t)
      nearest_hit = hit;
    if (sphere.is_light)
      light_sphere = &sphere;
  }
  if (!nearest_hit.is_hit)
    return Vec3();
  if (nearest_hit.sphere->is_light) {
    Vec3 Le = nearest_hit.sphere->color * 30.0f;
    if (prev_pdf == 0.0f)
      return Le;
    float p_light = light_pdf(*light_sphere, ray.origin, nearest_hit.position);
    float w = mis_weight(prev_pdf * rr_prob, p_light);
    return Le * w;
  }
  Vec3 wo = -ray.direction;
  Vec3 L_dir = Ldir(wo, light_sphere, nearest_hit, spheres);
  if (random_float() > rr_prob)
    return L_dir;
  Vec3 wi = sample_bsdf(nearest_hit.normal);
  float pdf_bsdf = bsdf_pdf(wi);
  Vec3 f_r_val = f_r(wi, wo, nearest_hit.sphere->color);
  float cos_theta = dot(wi, nearest_hit.normal);
  if (cos_theta <= 0.0f)
    return L_dir;
  Ray new_ray = Ray(nearest_hit.position + nearest_hit.normal * 1e-4f, wi);
  Vec3 Li = Lo(new_ray, spheres, depth - 1, pdf_bsdf);
  Vec3 L_indir = f_r_val * Li * cos_theta * (1.0f / (rr_prob * pdf_bsdf));
  return L_dir + L_indir;
}
int main() {
  int image_width = 1080, image_height = 1080;
  std::vector<Vec3> framebuffer(image_width * image_height);
  int samples_per_pixel = 1024;
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
        Ray ray = Ray(Vec3(), ray_direction);
        color = color + Lo(ray, spheres, 10, 0.0f);
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