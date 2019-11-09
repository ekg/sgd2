#include <random>
#include <iostream>
#include "sgd2.hpp"

void draw_svg(uint64_t n, double* X, uint64_t len_I, uint64_t* I, uint64_t* J) {
    double scale = 5.0;
    double border = 10.0;
    double min_x = 0;
    double min_y = 0;
    double max_x = 0;
    double max_y = 0;
    // determine boundaries
    for (uint64_t i = 0; i < n; ++i) {
        double x = X[i*2]*scale;
        double y = X[i*2+1]*scale;
        if (x < min_x) min_x = x;
        if (x > max_x) max_x = x;
        if (y < min_y) min_y = y;
        if (y > max_y) max_y = y;
    }
    double width = max_x - min_x;
    double height = max_y - min_y;
    
    std::cout << "<svg width=\"" << width + border << "\" height=\"" << height + border << "\" "
              << "viewBox=\"" << min_x - border/2<< " " << min_y - border/2 << " " << width + border << " " << height + border << "\" xmlns=\"http://www.w3.org/2000/svg\">"
              << "<style type=\"text/css\">"
              << "line{stroke:black;stroke-width:1.0;stroke-opacity:1.0;stroke-linecap:round;}"
        //<< "circle{{r:" << 1.0 << ";fill:black;fill-opacity:" << 1.0 << "}}"
              << "</style>"
              << std::endl;

    for (uint64_t i = 0; i < len_I; ++i) {
        uint64_t a = I[i];
        uint64_t b = J[i];
        std::cout << "<line x1=\"" << X[a*2]*scale << "\" x2=\"" << X[b*2]*scale
                  << "\" y1=\"" << X[a*2+1]*scale << "\" y2=\"" << X[b*2+1]*scale << "\"/>"
                  << std::endl;
    }

    /* // to draw nodes
    for (uint64_t i = 0; i < n; ++i) {
        std::cout << "<circle cx=\"" << X[i*2]*scale << "\" cy=\"" << X[i*2+1]*scale << "\" r=\"1.0\"/>" << std::endl;
    }
    */
    std::cout << "</svg>" << std::endl;
}

int main(void) {
    uint64_t n = 8;
    uint64_t len_I = 9;
    uint64_t I[len_I] = { 0,0,1,2,3,4,4,5,6 };
    uint64_t J[len_I] = { 1,2,3,3,4,5,6,7,7 };
    //double V[len_I] = { 1,2,1,1,3,1,2,2,1 };
    double X[2*n]; // = { 0,0, .1,.1, .2,.2, .3,.3, .4,.4, .5,.5, .6,.6, .7,.7 };
    std::random_device dev;
    // todo, seed with graph topology/contents to get a stable result
    std::mt19937 rng(dev());
    std::uniform_real_distribution<double> dist(0,1);
    for (uint64_t i = 0; i < 2*n; ++i) {
        X[i] = dist(rng);
    }
    //double X[2*n];
    uint64_t p = 10;
    uint64_t t_max = 30;
    double eps = 0.01;
    //layout_sparse_weighted(n, X, len_I, I, J, V, p, t_max, eps);
    layout_sparse_unweighted(n, X, len_I, I, J, p, t_max, eps);
    /*
    for (uint64_t i = 0; i < n; ++i) {
        std::cout << X[i*2] << " " << X[i*2+1] << std::endl;
    }
    */
    draw_svg(n, X, len_I, I, J);
    return 0;
}
