#define _CRT_SECURE_NO_WARNINGS 1
#include <vector>
#include <cmath>
#include <random>
#include <omp.h>
#include <fstream>
#include <map>
#include <string>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#ifndef M_PI
#define M_PI 3.14159265358979323856
#endif

static std::default_random_engine engine[32];
static std::uniform_real_distribution<double> uniform(0, 1);

double sqr(double x) { return x * x; };

class Vector {
public:
	explicit Vector(double x = 0, double y = 0, double z = 0) {
		data[0] = x;
		data[1] = y;
		data[2] = z;
	}
	double norm2() const {
		return data[0] * data[0] + data[1] * data[1] + data[2] * data[2];
	}
	double norm() const {
		return sqrt(norm2());
	}
	void normalize() {
		double n = norm();
		data[0] /= n;
		data[1] /= n;
		data[2] /= n;
	}
	double operator[](int i) const { return data[i]; };
	double& operator[](int i) { return data[i]; };
	double data[3];
};

Vector operator+(const Vector& a, const Vector& b) {
	return Vector(a[0] + b[0], a[1] + b[1], a[2] + b[2]);
}
Vector operator-(const Vector& a, const Vector& b) {
	return Vector(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}
Vector operator*(const double a, const Vector& b) {
	return Vector(a*b[0], a*b[1], a*b[2]);
}
Vector operator*(const Vector& a, const double b) {
	return Vector(a[0]*b, a[1]*b, a[2]*b);
}
Vector operator/(const Vector& a, const double b) {
	return Vector(a[0] / b, a[1] / b, a[2] / b);
}
double dot(const Vector& a, const Vector& b) {
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
Vector cross(const Vector& a, const Vector& b) {
	return Vector(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]);
}

Vector mult(const Vector& a, const Vector& b) {
	return Vector(a[0] * b[0], a[1] * b[1], a[2] * b[2]);
}

class Ray {
public:
	Ray(const Vector& origin, const Vector& unit_direction) : O(origin), u(unit_direction) {};
	Vector O, u;
};

class Object {
public:
	Object(const Vector& albedo, bool mirror = false, bool transparent = false) : albedo(albedo), mirror(mirror), transparent(transparent) {};

	virtual bool intersect(const Ray& ray, Vector& P, double& t, Vector& N) const = 0;

	Vector albedo;
	bool mirror, transparent;
};

class Sphere : public Object {
public:
	Sphere(const Vector& center, double radius, const Vector& albedo, bool mirror = false, bool transparent = false) : ::Object(albedo, mirror, transparent), C(center), R(radius) {};

	// returns true iif there is an intersection between the ray and the sphere
	// if there is an intersection, also computes the point of intersection P, 
	// t>=0 the distance between the ray origin and P (i.e., the parameter along the ray)
	// and the unit normal N
	bool intersect(const Ray& ray, Vector& P, double &t, Vector& N) const {
		// TODO (lab 1) : compute the intersection (just true/false at the begining of lab 1, then P, t and N as well)
		double delta = pow(dot(ray.u, ray.O-C),2) - ((ray.O-C).norm2() - pow(R,2));
		if (delta < 0){
			return false;
		}
		else if (delta == 0){
			double t0 = dot(ray.u, C-ray.O);
			if (t0 >= 0){
				t = t0;
				P = ray.O + t * ray.u;
				N = (P - C) / R;
				return true;
			}
		}
		else {
			double t1 = dot(ray.u, C-ray.O) - sqrt(delta);
			double t2 = dot(ray.u, C-ray.O) + sqrt(delta);
			if (t1 >= 0){
				t = t1;
				P = ray.O + t * ray.u;
				N = (P - C) / R;
				return true;
			}
			else if (t2 >= 0){
				t = t2;
				P = ray.O + t * ray.u;
				N = (P - C) / R;
				return true;
			}
		}

		return false;
	}

	double R;
	Vector C;
};


// Class only used in labs 3 and 4 
class TriangleIndices {
public:
	TriangleIndices(int vtxi = -1, int vtxj = -1, int vtxk = -1, int ni = -1, int nj = -1, int nk = -1, int uvi = -1, int uvj = -1, int uvk = -1, int group = -1) {
		vtx[0] = vtxi; vtx[1] = vtxj; vtx[2] = vtxk;
		uv[0] = uvi; uv[1] = uvj; uv[2] = uvk;
		n[0] = ni; n[1] = nj; n[2] = nk;
		this->group = group;
	};
	int vtx[3]; // indices within the vertex coordinates array
	int uv[3];  // indices within the uv coordinates array
	int n[3];   // indices within the normals array
	int group;  // face group
};

// Class only used in labs 3 and 4 
class TriangleMesh : public Object {
public:
	TriangleMesh(const Vector& albedo, bool mirror = false, bool transparent = false) : ::Object(albedo, mirror, transparent) {};

	// first scale and then translate the current object
	void scale_translate(double s, const Vector& t) {
		for (int i = 0; i < vertices.size(); i++) {
			vertices[i] = vertices[i] * s + t;
		}
	}

	// read an .obj file
	void readOBJ(const char* obj) {
		std::ifstream f(obj);
		if (!f) return;

		std::map<std::string, int> mtls;
		int curGroup = -1, maxGroup = -1;

		// OBJ indices are 1-based and can be negative (relative), this normalizes them
		auto resolveIdx = [](int i, int size) {
			return i < 0 ? size + i : i - 1;
		};

		auto setFaceVerts = [&](TriangleIndices& t, int i0, int i1, int i2) {
			t.vtx[0] = resolveIdx(i0, vertices.size());
			t.vtx[1] = resolveIdx(i1, vertices.size());
			t.vtx[2] = resolveIdx(i2, vertices.size());
		};
		auto setFaceUVs = [&](TriangleIndices& t, int j0, int j1, int j2) {
			t.uv[0] = resolveIdx(j0, uvs.size());
			t.uv[1] = resolveIdx(j1, uvs.size());
			t.uv[2] = resolveIdx(j2, uvs.size());
		};
		auto setFaceNormals = [&](TriangleIndices& t, int k0, int k1, int k2) {
			t.n[0] = resolveIdx(k0, normals.size());
			t.n[1] = resolveIdx(k1, normals.size());
			t.n[2] = resolveIdx(k2, normals.size());
		};

		std::string line;
		while (std::getline(f, line)) {
			// Trim trailing whitespace
			line.erase(line.find_last_not_of(" \r\t\n") + 1);
			if (line.empty()) continue;

			const char* s = line.c_str();

			if (line.rfind("usemtl ", 0) == 0) {
				std::string matname = line.substr(7);
				auto result = mtls.emplace(matname, maxGroup + 1);
				if (result.second) {
					curGroup = ++maxGroup;
				} else {
					curGroup = result.first->second;
				}
			} else if (line.rfind("vn ", 0) == 0) {
				Vector v;
				sscanf(s, "vn %lf %lf %lf", &v[0], &v[1], &v[2]);
				normals.push_back(v);
			} else if (line.rfind("vt ", 0) == 0) {
				Vector v;
				sscanf(s, "vt %lf %lf", &v[0], &v[1]);
				uvs.push_back(v);
			} else if (line.rfind("v ", 0) == 0) {
				Vector pos, col;
				if (sscanf(s, "v %lf %lf %lf %lf %lf %lf", &pos[0], &pos[1], &pos[2], &col[0], &col[1], &col[2]) == 6) {
					for (int i = 0; i < 3; i++) col[i] = std::min(1.0, std::max(0.0, col[i]));
					vertexcolors.push_back(col);
				} else {
					sscanf(s, "v %lf %lf %lf", &pos[0], &pos[1], &pos[2]);
				}
				vertices.push_back(pos);
			}
			else if (line[0] == 'f') {
				int i[4], j[4], k[4], offset, nn;
				const char* cur = s + 1;
				TriangleIndices t;
				t.group = curGroup;

				// Try each face format: v/vt/vn, v/vt, v//vn, v
				if ((nn = sscanf(cur, "%d/%d/%d %d/%d/%d %d/%d/%d%n", &i[0], &j[0], &k[0], &i[1], &j[1], &k[1], &i[2], &j[2], &k[2], &offset)) == 9) {
					setFaceVerts(t, i[0], i[1], i[2]); 
					setFaceUVs(t, j[0], j[1], j[2]); 
					setFaceNormals(t, k[0], k[1], k[2]);
				} else if ((nn = sscanf(cur, "%d/%d %d/%d %d/%d%n", &i[0], &j[0], &i[1], &j[1], &i[2], &j[2], &offset)) == 6) {
					setFaceVerts(t, i[0], i[1], i[2]); 
					setFaceUVs(t, j[0], j[1], j[2]);
				} else if ((nn = sscanf(cur, "%d//%d %d//%d %d//%d%n", &i[0], &k[0], &i[1], &k[1], &i[2], &k[2], &offset)) == 6) {
					setFaceVerts(t, i[0], i[1], i[2]); 
					setFaceNormals(t, k[0], k[1], k[2]);
				} else if ((nn = sscanf(cur, "%d %d %d%n", &i[0], &i[1], &i[2], &offset)) == 3) {
					setFaceVerts(t, i[0], i[1], i[2]);
				}
				else continue;

				indices.push_back(t);
				cur += offset;

				// Fan triangulation for polygon faces (4+ vertices)
				while (*cur && *cur != '\n') {
					TriangleIndices t2;
					t2.group = curGroup;
					if ((nn = sscanf(cur, " %d/%d/%d%n", &i[3], &j[3], &k[3], &offset)) == 3) {
						setFaceVerts(t2, i[0], i[2], i[3]); 
						setFaceUVs(t2, j[0], j[2], j[3]); 
						setFaceNormals(t2, k[0], k[2], k[3]);
					} else if ((nn = sscanf(cur, " %d/%d%n", &i[3], &j[3], &offset)) == 2) {
						setFaceVerts(t2, i[0], i[2], i[3]); 
						setFaceUVs(t2, j[0], j[2], j[3]);
					} else if ((nn = sscanf(cur, " %d//%d%n", &i[3], &k[3], &offset)) == 2) {
						setFaceVerts(t2, i[0], i[2], i[3]); 
						setFaceNormals(t2, k[0], k[2], k[3]);
					} else if ((nn = sscanf(cur, " %d%n", &i[3], &offset)) == 1) {
						setFaceVerts(t2, i[0], i[2], i[3]);
					} else { 
						cur++; 
						continue; 
					}

					indices.push_back(t2);
					cur += offset;
					i[2] = i[3]; j[2] = j[3]; k[2] = k[3];
				}
			}
		}
	}
	

	// TODO ray-mesh intersection (labs 3 and 4)
	bool intersect(const Ray& ray, Vector& P, double& t, Vector& N) const {
		
		// lab 3 : for each triangle, compute the ray-triangle intersection with Moller-Trumbore algorithm
		// lab 3 : once done, speed it up by first checking against the mesh bounding box
		// lab 4 : recursively apply the bounding-box test from a BVH datastructure
		if (vertices.empty() || indices.empty()) return false;

		if (!bvh_built) {
	#pragma omp critical
			{
				if (!bvh_built) build_bvh();
			}
		}

		if (bvh_nodes.empty()) return false;

		bool hit = false;
		double best_t = 1e30;
		Vector best_P;
		Vector best_N;

		intersect_bvh(0, ray, hit, best_t, best_P, best_N);

		if (!hit) return false;

		t = best_t;
		P = best_P;
		N = best_N;
		return true;
	}

	struct BVHNode {
		Vector bmin, bmax;
		int start, end;
		int left, right;
	};

	void bbox_reset(Vector& bmin, Vector& bmax) const {
		bmin = Vector(1e30, 1e30, 1e30);
		bmax = Vector(-1e30, -1e30, -1e30);
	}

	void bbox_add_point(Vector& bmin, Vector& bmax, const Vector& p) const {
		for (int a = 0; a < 3; ++a) {
			if (p[a] < bmin[a]) bmin[a] = p[a];
			if (p[a] > bmax[a]) bmax[a] = p[a];
		}
	}

	Vector triangle_center(int tri_id) const {
		const TriangleIndices& tri = indices[tri_id];
		return (vertices[tri.vtx[0]] + vertices[tri.vtx[1]] + vertices[tri.vtx[2]]) / 3.0;
	}

	void triangle_bbox(int tri_id, Vector& bmin, Vector& bmax) const {
		const TriangleIndices& tri = indices[tri_id];

		bbox_reset(bmin, bmax);

		bbox_add_point(bmin, bmax, vertices[tri.vtx[0]]);
		bbox_add_point(bmin, bmax, vertices[tri.vtx[1]]);
		bbox_add_point(bmin, bmax, vertices[tri.vtx[2]]);
	}

	double triangle_center_axis(int tri_id, int axis) const {
		const TriangleIndices& tri = indices[tri_id];

		return (
			vertices[tri.vtx[0]][axis] +
			vertices[tri.vtx[1]][axis] +
			vertices[tri.vtx[2]][axis]
		) / 3.0;
	}

	void swap_triangles(int i, int j) const {
		int tmp = bvh_triangles[i];
		bvh_triangles[i] = bvh_triangles[j];
		bvh_triangles[j] = tmp;
	}

	void sort_triangles(int left, int right, int axis) const {
		int i = left;
		int j = right;

		double pivot = triangle_center_axis(bvh_triangles[(left + right) / 2], axis);

		while (i <= j) {
			while (triangle_center_axis(bvh_triangles[i], axis) < pivot) ++i;
			while (triangle_center_axis(bvh_triangles[j], axis) > pivot) --j;

			if (i <= j) {
				swap_triangles(i, j);
				++i;
				--j;
			}
		}

		if (left < j) sort_triangles(left, j, axis);
		if (i < right) sort_triangles(i, right, axis);
	}

	void build_bvh() const {
		bvh_nodes.clear();
		bvh_triangles.resize(indices.size());

		for (int i = 0; i < (int)indices.size(); ++i) {
			bvh_triangles[i] = i;
		}

		if (!indices.empty()) {
			build_bvh_node(0, (int)indices.size());
		}

		bvh_built = true;
	}

	int build_bvh_node(int start, int end) const {
		BVHNode node;

		node.start = start;
		node.end = end;
		node.left = -1;
		node.right = -1;

		bbox_reset(node.bmin, node.bmax);

		Vector centroid_min;
		Vector centroid_max;
		bbox_reset(centroid_min, centroid_max);

		for (int i = start; i < end; ++i) {
			Vector tri_min;
			Vector tri_max;

			triangle_bbox(bvh_triangles[i], tri_min, tri_max);

			bbox_add_point(node.bmin, node.bmax, tri_min);
			bbox_add_point(node.bmin, node.bmax, tri_max);

			bbox_add_point(centroid_min, centroid_max, triangle_center(bvh_triangles[i]));
		}

		int node_id = (int)bvh_nodes.size();
		bvh_nodes.push_back(node);

		int count = end - start;

		if (count <= 4) {
			return node_id;
		}

		Vector extent = centroid_max - centroid_min;

		int axis = 0;
		if (extent[1] > extent[axis]) axis = 1;
		if (extent[2] > extent[axis]) axis = 2;

		if (extent[axis] < 1e-12) {
			return node_id;
		}

		sort_triangles(start, end - 1, axis);

		int mid = (start + end) / 2;

		bvh_nodes[node_id].left = build_bvh_node(start, mid);
		bvh_nodes[node_id].right = build_bvh_node(mid, end);

		return node_id;
	}

	bool bbox_intersect(
		const Ray& ray,
		const Vector& bmin,
		const Vector& bmax,
		double max_t,
		double& tnear
	) const {
		double tmin = 0.0;
		double tmax = max_t;

		for (int a = 0; a < 3; ++a) {
			if (std::fabs(ray.u[a]) < 1e-12) {
				if (ray.O[a] < bmin[a] || ray.O[a] > bmax[a]) {
					return false;
				}
			} else {
				double t0 = (bmin[a] - ray.O[a]) / ray.u[a];
				double t1 = (bmax[a] - ray.O[a]) / ray.u[a];

				if (t0 > t1) {
					double tmp = t0;
					t0 = t1;
					t1 = tmp;
				}

				if (t0 > tmin) tmin = t0;
				if (t1 < tmax) tmax = t1;

				if (tmin > tmax) return false;
			}
		}

		tnear = tmin;
		return true;
	}

	bool intersect_triangle(int tri_id, const Ray& ray, Vector& P, double& t, Vector& N) const {
		const TriangleIndices& tri = indices[tri_id];

		const Vector& A = vertices[tri.vtx[0]];
		const Vector& B = vertices[tri.vtx[1]];
		const Vector& C = vertices[tri.vtx[2]];

		Vector e1 = B - A;
		Vector e2 = C - A;

		Vector pvec = cross(ray.u, e2);
		double det = dot(e1, pvec);

		if (std::fabs(det) < 1e-8) return false;

		double invDet = 1.0 / det;

		Vector tvec = ray.O - A;
		double beta = dot(tvec, pvec) * invDet;

		if (beta < 0.0 || beta > 1.0) return false;

		Vector qvec = cross(tvec, e1);
		double gamma = dot(ray.u, qvec) * invDet;

		if (gamma < 0.0 || beta + gamma > 1.0) return false;

		double cur_t = dot(e2, qvec) * invDet;

		if (cur_t < 1e-8) return false;

		t = cur_t;
		P = ray.O + cur_t * ray.u;

		double alpha = 1.0 - beta - gamma;

		if (tri.n[0] >= 0 && tri.n[1] >= 0 && tri.n[2] >= 0 && !normals.empty()) {
			N =
				alpha * normals[tri.n[0]]
				+ beta * normals[tri.n[1]]
				+ gamma * normals[tri.n[2]];
		} else {
			N = cross(e1, e2);
		}

		N.normalize();

		if (dot(N, ray.u) > 0) {
			N = -1.0 * N;
		}

		return true;
	}

	void intersect_bvh(
		int node_id,
		const Ray& ray,
		bool& hit,
		double& best_t,
		Vector& best_P,
		Vector& best_N
	) const {
		const BVHNode& node = bvh_nodes[node_id];

		double node_t;

		if (!bbox_intersect(ray, node.bmin, node.bmax, best_t, node_t)) {
			return;
		}

		if (node.left < 0 && node.right < 0) {
			for (int i = node.start; i < node.end; ++i) {
				Vector cur_P;
				Vector cur_N;
				double cur_t;

				if (
					intersect_triangle(bvh_triangles[i], ray, cur_P, cur_t, cur_N)
					&& cur_t < best_t
				) {
					hit = true;
					best_t = cur_t;
					best_P = cur_P;
					best_N = cur_N;
				}
			}

			return;
		}

		double left_t = 1e30;
		double right_t = 1e30;

		bool hit_left =
			node.left >= 0
			&& bbox_intersect(
				ray,
				bvh_nodes[node.left].bmin,
				bvh_nodes[node.left].bmax,
				best_t,
				left_t
			);

		bool hit_right =
			node.right >= 0
			&& bbox_intersect(
				ray,
				bvh_nodes[node.right].bmin,
				bvh_nodes[node.right].bmax,
				best_t,
				right_t
			);

		if (hit_left && hit_right) {
			if (left_t < right_t) {
				intersect_bvh(node.left, ray, hit, best_t, best_P, best_N);
				intersect_bvh(node.right, ray, hit, best_t, best_P, best_N);
			} else {
				intersect_bvh(node.right, ray, hit, best_t, best_P, best_N);
				intersect_bvh(node.left, ray, hit, best_t, best_P, best_N);
			}
		} else if (hit_left) {
			intersect_bvh(node.left, ray, hit, best_t, best_P, best_N);
		} else if (hit_right) {
			intersect_bvh(node.right, ray, hit, best_t, best_P, best_N);
		}
	}

	std::vector<TriangleIndices> indices;
	std::vector<Vector> vertices;
	std::vector<Vector> normals;
	std::vector<Vector> uvs;
	std::vector<Vector> vertexcolors;

	mutable std::vector<int> bvh_triangles;
	mutable std::vector<BVHNode> bvh_nodes;
	mutable bool bvh_built = false;
};

class Scene {
public:
	Scene() {};
	void addObject(const Object* obj) {
		objects.push_back(obj);
	}

	// returns true iif there is an intersection between the ray and any object in the scene
    // if there is an intersection, also computes the point of the *nearest* intersection P, 
    // t>=0 the distance between the ray origin and P (i.e., the parameter along the ray)
    // and the unit normal N. 
	// Also returns the index of the object within the std::vector objects in object_id
	bool intersect(const Ray& ray, Vector& P, double& t, Vector& N, int &object_id) const  {	
		// TODO (lab 1): iterate through the objects and check the intersections with all of them, 
		// and keep the closest intersection, i.e., the one if smallest positive value of t
		bool val = false;
		for (size_t i = 0; i < objects.size(); ++i){
			double tempt{};
			Vector tempP{};
			Vector tempN{};
			if (objects[i]->intersect(ray, tempP, tempt, tempN)){
				val = true;
				if (tempt <= t){
					t = tempt;
					P = tempP;
					N = tempN;
					object_id = i;
				}
			}
		}
		return val;
	}


	// return the radiance (color) along ray
	Vector getColor(const Ray& ray, int recursion_depth) {

		if (recursion_depth >= max_light_bounce) return Vector(0, 0, 0);

		// TODO (lab 1) : if intersect with ray, use the returned information to compute the color ; otherwise black 
		// in lab 1, the color only includes direct lighting with shadows

		Vector P, N; 
		double t = 1e10;
		int object_id;
		if (intersect(ray, P, t, N, object_id)) {

			if (objects[object_id]->mirror) {
				Ray reflRay(P + 1e-5 * N, ray.u - 2*dot(ray.u, N) * N);
				return getColor(reflRay, recursion_depth + 1);
				// return getColor in the reflected direction, with recursion_depth+1 (recursively)
			} // else

			if (objects[object_id]->transparent) { // optional

				// return getColor in the refraction direction, with recursion_depth+1 (recursively)
			} // else

			Vector direct(0., 0., 0.);
			Vector indirect(0., 0., 0.);

			Vector to_light = light_position - P;
			double dist_to_light = to_light.norm();
			Vector light_dir = to_light / dist_to_light;

			Ray shadowRay(P + 1e-5 * N, light_dir);

			Vector shadowP, shadowN;
			double shadowT = 1e10;
			int shadowObjectId;

			if (!(intersect(shadowRay, shadowP, shadowT, shadowN, shadowObjectId) && shadowT < dist_to_light)) {
				double cos_theta = std::max(0.0, dot(N, light_dir));
				direct =
					(light_intensity / (4.0 * M_PI * dist_to_light * dist_to_light)) *
					cos_theta *
					(objects[object_id]->albedo / M_PI);
			}

			int thread_id = omp_get_thread_num();

			double r1 = uniform(engine[thread_id]);
			double r2 = uniform(engine[thread_id]);

			double x = std::cos(2.0 * M_PI * r1) * std::sqrt(1.0 - r2);
			double y = std::sin(2.0 * M_PI * r1) * std::sqrt(1.0 - r2);
			double z = std::sqrt(r2);

			Vector T1;
			if (std::fabs(N[0]) < std::fabs(N[1])) {
				T1 = cross(Vector(1, 0, 0), N);
			} else {
				T1 = cross(Vector(0, 1, 0), N);
			}
			T1.normalize();
			Vector T2 = cross(N, T1);

			Vector wi = x * T1 + y * T2 + z * N;
			wi.normalize();

			Ray indirectRay(P + 1e-5 * N, wi);
			Vector Li = getColor(indirectRay, recursion_depth + 1);

			indirect = mult(objects[object_id]->albedo, Li);

			return direct + indirect;

		}

		return Vector(0, 0, 0);
	}

	std::vector<const Object*> objects;

	Vector camera_center, light_position;
	double fov, gamma, light_intensity;
	int max_light_bounce;
};


int main() {
	int W = 512;
	int H = 512;

	for (int i = 0; i<32; i++) {
		engine[i].seed(i);
	}

	Sphere center_sphere(Vector(0, 0, 0), 10., Vector(0.8, 0.8, 0.8), true);
	Sphere wall_left(Vector(-1000, 0, 0), 940, Vector(0.5, 0.8, 0.1));
	Sphere wall_right(Vector(1000, 0, 0), 940, Vector(0.9, 0.2, 0.3));
	Sphere wall_front(Vector(0, 0, -1000), 940, Vector(0.1, 0.6, 0.7));
	Sphere wall_behind(Vector(0, 0, 1000), 940, Vector(0.8, 0.2, 0.9));
	Sphere ceiling(Vector(0, 1000, 0), 940, Vector(0.3, 0.5, 0.3));
	Sphere floor(Vector(0, -1000, 0), 990, Vector(0.6, 0.5, 0.7));

	Scene scene;
	scene.camera_center = Vector(0, 0, 55);
	scene.light_position = Vector(-10,20,40);
	scene.light_intensity = 1E7;
	scene.fov = 60 * M_PI / 180.;
	scene.gamma = 2.2;    // TODO (lab 1) : play with gamma ; typically, gamma = 2.2
	scene.max_light_bounce = 5;

	TriangleMesh cat(Vector(0.8,0.8, 0.8));
	cat.readOBJ("cat.obj");
	cat.scale_translate(0.6, Vector(0, -10, 0));
	//scene.addObject(&center_sphere);

	scene.addObject(&wall_left);
	scene.addObject(&wall_right);
	scene.addObject(&wall_front);
	scene.addObject(&wall_behind);
	scene.addObject(&ceiling);
	scene.addObject(&floor);

	scene.addObject(&cat);
	

	std::vector<unsigned char> image(W * H * 3, 0);

#pragma omp parallel for schedule(dynamic, 1)
for (int i = 0; i < H; i++) {
    for (int j = 0; j < W; j++) {
        Vector color(0., 0., 0.);

        int nb_paths = 128;
        double sigma = 0.5;
        int thread_id = omp_get_thread_num();

        for (int k = 0; k < nb_paths; ++k) {
            double r1 = uniform(engine[thread_id]);
            double r2 = uniform(engine[thread_id]);
            r1 = std::max(r1, 1e-12);

            double gx = sigma * std::sqrt(-2.0 * std::log(r1)) * std::cos(2.0 * M_PI * r2);
            double gy = sigma * std::sqrt(-2.0 * std::log(r1)) * std::sin(2.0 * M_PI * r2);

            double px = j + 0.5 + gx;
            double py = i + 0.5 + gy;

            Vector ray_direction(
                px - W * 0.5,
                H * 0.5 - py,
                -(W / (2.0 * std::tan(scene.fov * 0.5)))
            );
            ray_direction.normalize();

            Ray ray(scene.camera_center, ray_direction);
            color = color + scene.getColor(ray, 0);
        }

        color = color / double(nb_paths);

        image[(i * W + j) * 3 + 0] = std::min(255., std::max(0., 255. * std::pow(color[0] / 255., 1. / scene.gamma)));
        image[(i * W + j) * 3 + 1] = std::min(255., std::max(0., 255. * std::pow(color[1] / 255., 1. / scene.gamma)));
        image[(i * W + j) * 3 + 2] = std::min(255., std::max(0., 255. * std::pow(color[2] / 255., 1. / scene.gamma)));
    }
}
	stbi_write_png("image.png", W, H, 3, &image[0], 0);

	return 0;
} 