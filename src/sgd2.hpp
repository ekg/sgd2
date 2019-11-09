#pragma once

#include <vector>
#include <cstdint>
#include <algorithm>
#include <queue>
#include <unordered_set>
#include <unordered_map>
#include <limits>
#include <cmath>
#include <chrono>
#include <iostream>
#include <vector>
#include <algorithm>
#include <queue>
#include <unordered_set>
#include <unordered_map>
#include <random>

namespace sgd2 {

////////////////////////
// External interface //
////////////////////////
void layout_unweighted(uint64_t n, double* X, uint64_t m, uint64_t* I, uint64_t* J, uint64_t t_max, double eps);
void layout_weighted(uint64_t n, double* X, uint64_t m, uint64_t* I, uint64_t* J, double* V, uint64_t t_max, double eps);
void layout_unweighted_convergent(uint64_t n, double* X, uint64_t m, uint64_t* I, uint64_t* J, uint64_t t_max, double eps, double delta, uint64_t t_maxmax);
void layout_weighted_convergent(uint64_t n, double* X, uint64_t m, uint64_t* I, uint64_t* J, double* V, uint64_t t_max, double eps, double delta, uint64_t t_maxmax);

void layout_sparse_unweighted(uint64_t n, double* X, uint64_t m, uint64_t* I, uint64_t* J, uint64_t p, uint64_t t_max, double eps);
void layout_sparse_weighted(uint64_t n, double* X, uint64_t m, uint64_t* I, uint64_t* J, double* V, uint64_t p, uint64_t t_max, double eps);

void mds_direct(uint64_t n, double* X, double* d, double* w, uint64_t t_max, double* etas);


struct term
{
    uint64_t i, j;
    double d, w;
    term(uint64_t i, uint64_t j, double d, double w) : i(i), j(j), d(d), w(w) {}
};
void sgd(double* X, std::vector<term> &terms, const std::vector<double> &etas, const double delta=0);

// for Dijkstra
struct edge
{
    // NOTE: this will be used for 'invisible' edges in the Dijkstra priority queue
    uint64_t target;
    double weight;
    edge(uint64_t target, double weight) : target(target), weight(weight) {}
};

struct edge_comp
{
    bool operator() (const edge &lhs, const edge &rhs) const
    {
        return lhs.weight > rhs.weight;
    }
};

// Ortmann et al. stuff
struct term_sparse
{
    uint64_t i, j;
    double d, w_ij, w_ji;
    term_sparse(uint64_t i, uint64_t j, double d) : i(i), j(j), d(d), w_ij(0), w_ji(0) {}
};

std::vector<std::vector<uint64_t>> build_graph_unweighted(uint64_t n, uint64_t m, uint64_t* I, uint64_t* J);
std::vector<std::vector<edge>> build_graph_weighted(uint64_t n, uint64_t m, uint64_t* I, uint64_t* J, double* V);
std::vector<term> bfs(uint64_t n, uint64_t m, uint64_t* I, uint64_t* J);
std::vector<term> dijkstra(uint64_t n, uint64_t m, uint64_t* I, uint64_t* J, double* V);

std::vector<double> schedule(const std::vector<term> &terms, uint64_t t_max, double eps);
std::vector<double> schedule_convergent(const std::vector<term> &terms, uint64_t t_max, double eps, uint64_t t_maxmax);


void sgd(double* X, std::vector<term_sparse>& terms, const std::vector<double>& etas);

std::vector<uint64_t> maxmin_random_sp_unweighted(const std::vector<std::vector<uint64_t>>& graph, uint64_t n_pivots, uint64_t p0 = 0);
std::vector<uint64_t> maxmin_random_sp_weighted(const std::vector<std::vector<edge>>& graph, uint64_t n_pivots, uint64_t p0 = 0);
void maxmin_bfs_unweighted(const std::vector<std::vector<uint64_t>>& graph, const uint64_t p, std::vector<uint64_t>& mins, std::vector<uint64_t>& argmins);
void maxmin_bfs_weighted(const std::vector<std::vector<edge>>& graph, const uint64_t p, std::vector<double>& mins, std::vector<uint64_t>& argmins);
std::vector<term_sparse> MSSP_unweighted(const std::vector<std::vector<uint64_t>>& graph, const std::vector<uint64_t>& pivots);
std::vector<term_sparse> MSSP_weighted(const std::vector<std::vector<edge>>& graph, const std::vector<uint64_t>& pivots);

//#include <cmath>
//#include "layout.hpp"

void sgd(double* X, std::vector<term> &terms, const std::vector<double> &etas, double delta)
{
    // iterate through step sizes
    for (double eta : etas)
    {
        // shuffle terms
        std::random_shuffle(terms.begin(), terms.end());

        double Delta_max = 0;
        for (const term &t : terms)
        {
            // cap step size
            double w_ij = t.w;
            double mu = eta * w_ij;
            if (mu > 1)
                mu = 1;

            double d_ij = t.d;
            uint64_t i = t.i, j = t.j;

            double dx = X[i*2]-X[j*2], dy = X[i*2+1]-X[j*2+1];
            double mag = sqrt(dx*dx + dy*dy);

            // check distances for early stopping
            double Delta = mu * (mag-d_ij) / 2;
            if (Delta > Delta_max)
                Delta_max = Delta;

            double r = Delta / mag;
            double r_x = r * dx;
            double r_y = r * dy;
            
            X[i*2] -= r_x;
            X[i*2+1] -= r_y;
            X[j*2] += r_x;
            X[j*2+1] += r_y;
        }
        //std::cerr << ++iteration << ", eta: " << eta << ", Delta: " << Delta_max << std::endl;
        if (Delta_max < delta)
            return;
    }
}

std::vector<std::vector<uint64_t>> build_graph_unweighted(uint64_t n, uint64_t m, uint64_t* I, uint64_t* J)
{
    // used to make graph undirected, in case it is not already
    std::vector<std::unordered_set<uint64_t>> undirected(n);
    std::vector<std::vector<uint64_t>> graph(n);

    for (uint64_t ij=0; ij<m; ij++)
    {
        uint64_t i = I[ij], j = J[ij];
        if (i >= n || j >= n)
            throw "i or j bigger than n";

        if (i != j && undirected[j].find(i) == undirected[j].end()) // if edge not seen
        {
            undirected[i].insert(j);
            undirected[j].insert(i);
            graph[i].push_back(j);
            graph[j].push_back(i);
        }
    }
    return graph;
}

// calculates the unweighted shortest paths between indices I and J
// using a breadth-first search, returning a vector of terms
std::vector<term> bfs(uint64_t n, uint64_t m, uint64_t* I, uint64_t* J)
{
    auto graph = build_graph_unweighted(n, m, I, J);

    uint64_t nC2 = (n*(n-1))/2;
    std::vector<term> terms;
    terms.reserve(nC2);

    uint64_t terms_size_goal = 0; // to keep track of when to stop searching i<j

    for (uint64_t source=0; source<n-1; source++) // no need to do final vertex because i<j
    {
        std::vector<uint64_t> d(n, -1); // distances from source
        std::queue<uint64_t> q;
        
        d[source] = 0;
        q.push(source);

        terms_size_goal += n-source-1; // this is how many terms exist for i<j

        while (!q.empty() && terms.size() <= terms_size_goal)
        {
            uint64_t current = q.front();
            q.pop();
            for (uint64_t next : graph[current])
            {
                if (d[next] == -1)
                {
                    q.push(next);
                    uint64_t d_ij = d[current] + 1;
                    d[next] = d_ij;

                    if (source < next) // only add terms for i<j
                    {
                        double w_ij = 1.0 / ((double)d_ij*d_ij);
                        terms.push_back(term(source, next, d_ij, w_ij));
                    }
                }
            }
        }
        if (terms.size() != terms_size_goal)
        {
            throw "graph is not strongly connected, or is not indexed from zero";
        }
    }
    return terms;
}


std::vector<std::vector<edge>> build_graph_weighted(uint64_t n, uint64_t m, uint64_t* I, uint64_t* J, double* V)
{
    // used to make graph undirected, in case graph is not already
    std::vector<std::unordered_map<uint64_t, double>> undirected(n);
    std::vector<std::vector<edge>> graph(n);

    for (uint64_t ij=0; ij<m; ij++)
    {
        uint64_t i = I[ij], j = J[ij];
        if (i >= n || j >= n)
            throw "i or j bigger than n";

        double v = V[ij];
        if (v <= 0)
            throw "v less or equal 0";

        if (i != j && undirected[j].find(i) == undirected[j].end()) // if key not there
        {
            undirected[i].insert({j, v});
            undirected[j].insert({i, v});
            graph[i].push_back(edge(j, v));
            graph[j].push_back(edge(i, v));
        }
        else
        {
            if (undirected[j][i] != v)
                throw "graph weights not symmetric";
        }
    }
    return graph;
}

// calculates the unweighted shortest paths between indices I and J
// using Dijkstra's algorithm, returning a vector of terms
std::vector<term> dijkstra(uint64_t n, uint64_t m, uint64_t* I, uint64_t* J, double* V)
{
    auto graph = build_graph_weighted(n, m, I, J, V);

    uint64_t nC2 = (n*(n-1))/2;
    std::vector<term> terms;
    terms.reserve(nC2);

    uint64_t terms_size_goal = 0; // to keep track of when to stop searching i<j

    for (uint64_t source=0; source<n-1; source++) // no need to do final vertex because i<j
    {
        std::vector<bool> visited(n, false);
        std::vector<double> d(n, std::numeric_limits<double>::max()); // init 'tentative' distances to infinity

        // I am not using a fibonacci heap. I AM NOT USING A FIBONACCI HEAP
        // edges are used 'invisibly' here
        std::priority_queue<edge, std::vector<edge>, edge_comp> pq;

        d[source] = 0;
        pq.push(edge(source,0));

        terms_size_goal += n-source-1; // this is how many terms exist for i<j

        while (!pq.empty() && terms.size() <= terms_size_goal)
        {
            uint64_t current = pq.top().target;
            double d_ij = pq.top().weight;
            pq.pop();
            
            if (!visited[current]) // ignore redundant elements in queue
            {
                visited[current] = true;

                if (source < current) // only add terms for i<j
                {
                    double w_ij = 1.0 / (d_ij*d_ij);
                    terms.push_back(term(source, current, d_ij, w_ij));
                }
                for (edge e : graph[current])
                {
                    // here the edge is not 'invisible'
                    uint64_t next = e.target;
                    double weight = e.weight;

                    if (d_ij + weight < d[next]) // update tentative value of d 
                    {
                        d[next] = d_ij + weight;
                        pq.push(edge(next, d[next]));
                    }
                }
            }
        }
        if (terms.size() != terms_size_goal)
        {
            throw "graph is not strongly connected, or is not indexed from zero";
        }
    }
    return terms;
}


std::vector<double> schedule(const std::vector<term> &terms, uint64_t t_max, double eps)
{
    double w_min = terms[0].w, w_max = terms[0].w;
    for (uint64_t i=1; i<terms.size(); i++)
    {
        double w = terms[i].w;
        if (w < w_min) w_min = w;
        if (w > w_max) w_max = w;
    }
    double eta_max = 1.0 / w_min;
    double eta_min = eps / w_max;

    double lambda = log(eta_max/eta_min) / ((double)t_max-1);

    // initialize step sizes
    std::vector<double> etas;
    etas.reserve(t_max);
    for (uint64_t t=0; t<t_max; t++)
        etas.push_back(eta_max * exp(-lambda * t));

    return etas;
}

std::vector<double> schedule_convergent(const std::vector<term> &terms, uint64_t t_max, double eps, uint64_t t_maxmax)
{
    double w_min = terms[0].w, w_max = terms[0].w;
    for (uint64_t i=1; i<terms.size(); i++)
    {
        double w = terms[i].w;
        if (w < w_min) w_min = w;
        if (w > w_max) w_max = w;
    }
    double eta_max = 1.0 / w_min;
    double eta_min = eps / w_max;

    double lambda = log(eta_max/eta_min) / ((double)t_max-1);

    // initialize step sizes
    std::vector<double> etas;
    etas.reserve(t_maxmax);
    double eta_switch = 1.0 / w_max;
    for (uint64_t t=0; t<t_maxmax; t++)
    {
        double eta = eta_max * exp(-lambda * t);
        if (eta < eta_switch)
            break;

        etas.push_back(eta);
    }
    uint64_t tau = etas.size();
    for (uint64_t t=tau; t<t_maxmax; t++)
    {
        double eta = eta_switch / (1 + lambda*((double)t-tau));
        etas.push_back(eta);
    }

    return etas;
}

void layout_unweighted(uint64_t n, double* X, uint64_t m, uint64_t* I, uint64_t* J, uint64_t t_max, double eps)
{
    try
    {
        //auto start = std::chrono::steady_clock::now();

        std::vector<term> terms = bfs(n, m, I, J);
        //auto end = std::chrono::steady_clock::now();
        //std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() << "ms" << std::endl;

        std::vector<double> etas = schedule(terms, t_max, eps);
        sgd(X, terms, etas);
        //end = std::chrono::steady_clock::now();
        //std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() << "ms" << std::endl;
    }
    catch (const char* msg)
    {
        std::cerr << "Error: " << msg << std::endl;
    }
}

void layout_weighted(uint64_t n, double* X, uint64_t m, uint64_t* I, uint64_t* J, double* V, uint64_t t_max, double eps)
{
    try
    {
        std::vector<term> terms = dijkstra(n, m, I, J, V);
        std::vector<double> etas = schedule(terms, t_max, eps);
        sgd(X, terms, etas);
    }
    catch (const char* msg)
    {
        std::cerr << "Error: " << msg << std::endl;
    }
}
void layout_unweighted_convergent(uint64_t n, double* X, uint64_t m, uint64_t* I, uint64_t* J, uint64_t t_max, double eps, double delta, uint64_t t_maxmax)
{
    try
    {
        std::vector<term> terms = bfs(n, m, I, J);
        std::vector<double> etas = schedule_convergent(terms, t_max, eps, t_maxmax);
        sgd(X, terms, etas, delta);
    } 
    catch (const char* msg)
    {
        std::cerr << "Error: " << msg << std::endl;
    }
}
void layout_weighted_convergent(uint64_t n, double* X, uint64_t m, uint64_t* I, uint64_t* J, double* V, uint64_t t_max, double eps, double delta, uint64_t t_maxmax)
{
    try
    {
        std::vector<term> terms = dijkstra(n, m, I, J, V);
        std::vector<double> etas = schedule_convergent(terms, t_max, eps, t_maxmax);
        sgd(X, terms, etas, delta);
    }
    catch (const char* msg)
    {
        std::cerr << "Error: " << msg << std::endl;
    }
}

// d and w should be condensed distance matrices
void mds_direct(uint64_t n, double* X, double* d, double* w, uint64_t t_max, double* eta)
{
    // initialize SGD
    uint64_t nC2 = (n*(n-1))/2;
    std::vector<term> terms;
    terms.reserve(nC2);
    uint64_t ij=0;
    for (uint64_t i=0; i<n; i++) // unpack the condensed distance matrices
    {
        for (uint64_t j=i+1; j<n; j++)
        {
            terms.push_back(term(i, j, d[ij], w[ij]));
            ij += 1;
        }
    }

    // initialize step sizes
    std::vector<double> etas;
    etas.reserve(t_max);
    for (uint64_t t=0; t<t_max; t++)
    {
        etas.push_back(eta[t]);
    }
    
    sgd(X, terms, etas, 0);
}

//#include <cmath>

//#include "layout.hpp"

void sgd(double* X, std::vector<term_sparse> &terms, const std::vector<double> &etas)
{
    // iterate through step sizes
    for (double eta : etas)
    {
        // shuffle terms
        std::random_shuffle(terms.begin(), terms.end());

        for (const term_sparse& t : terms)
        {
            // cap step size
            double mu_i = eta * t.w_ij;
            if (mu_i > 1)
                mu_i = 1;

            // cap step size
            double mu_j = eta * t.w_ji;
            if (mu_j > 1)
                mu_j = 1;

            double d_ij = t.d;
            uint64_t i = t.i, j = t.j;

            double dx = X[i*2]-X[j*2], dy = X[i*2+1]-X[j*2+1];
            double mag = sqrt(dx*dx + dy*dy);

            double r = (mag-d_ij) / (2*mag);
            double r_x = r * dx;
            double r_y = r * dy;

            X[i*2] -= mu_i * r_x;
            X[i*2+1] -= mu_i * r_y;
            X[j*2] += mu_j * r_x;
            X[j*2+1] += mu_j * r_y;
        }
    }
}

// returns closest pivot for each vertex, not the pivots themselves
std::vector<uint64_t> maxmin_random_sp_unweighted(const std::vector<std::vector<uint64_t>>& graph, uint64_t n_pivots, uint64_t p0)
{
    uint64_t n = graph.size();

    std::vector<uint64_t> mins(n, std::numeric_limits<uint64_t>::max());
    std::vector<uint64_t> argmins(n, -1);

    // first pivot
    mins[p0] = 0;
    argmins[p0] = p0;
    maxmin_bfs_unweighted(graph, p0, mins, argmins);
    for (uint64_t i = 0; i < n; i++)
    {
        if (argmins[i] == -1)
            throw "graph is not strongly connected, or is not indexed from zero";
    }

    // remaining pivots
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<> rng(0, 1);
    for (uint64_t i = 1; i < n_pivots; i++)
    {
        // uint64_t max = mins[0], argmax = 0;
        // for (uint64_t i = 1; i < n; i++)
        // {
        //     if (mins[i] > max)
        //     {
        //         max = mins[i];
        //         argmax = i;
        //     }
        // }
        // maxmin non-random above

        // choose pivots with probability min
        uint64_t min_total = 0;
        for (uint64_t i = 0; i < n; i++)
        {
            min_total += mins[i];
        }
        double rn = rng(mt) * min_total;
        uint64_t cumul = 0;
        uint64_t argmax = 0;
        for (uint64_t i = 1; i < n; i++)
        {
            cumul += mins[i];
            if (cumul >= rn)
            {
                argmax = i;
                break;
            }
        }

        mins[argmax] = 0;
        argmins[argmax] = argmax;
        maxmin_bfs_unweighted(graph, argmax, mins, argmins);
    }
    // TODO: look for error in bfs here
    return argmins;
}
void maxmin_bfs_unweighted(const std::vector<std::vector<uint64_t>>& graph, const uint64_t p, std::vector<uint64_t>& mins, std::vector<uint64_t>& argmins)
{
    uint64_t n = graph.size();
    std::queue<uint64_t> q;
    std::vector<uint64_t> d(n, -1);

    q.push(p);
    d[p] = 0;
    while (!q.empty())
    {
        uint64_t current = q.front();
        q.pop();
        for (uint64_t next : graph[current])
        {
            if (d[next] == -1)
            {
                q.push(next);
                d[next] = d[current] + 1;
                if (d[next] < mins[next])
                {
                    mins[next] = d[next];
                    argmins[next] = p;
                }
            }
        }
    }
}

// is not actually a multi-source shortest path, because regions come for free with maxmin_random_sp
std::vector<term_sparse> MSSP_unweighted(const std::vector<std::vector<uint64_t>>& graph, const std::vector<uint64_t>& closest_pivots)
{
    uint64_t n = graph.size();

    // get pivots and their regions, but in sets
    std::unordered_map<uint64_t, std::unordered_set<uint64_t>> regions;
    for (uint64_t i = 0; i < n; i++)
    {
        if (regions.find(closest_pivots[i]) == regions.end())
        {
            regions[closest_pivots[i]] = std::unordered_set<uint64_t>();
        }
        regions[closest_pivots[i]].insert(i);
    }

    std::unordered_map<uint64_t, std::unordered_map<uint64_t, term_sparse>> termsDict;
    for (const auto& region : regions)
    {
        // q contains next to visit
        std::queue<uint64_t> q;
        std::vector<uint64_t> d(n, -1);

        uint64_t p = region.first;
        q.push(p);
        d[p] = 0;

        // q2 contains visited vertices' distances for s calculation
        std::queue<uint64_t> q2;
        uint64_t s = 0;
        q2.push(0);

        while (!q.empty())
        {
            uint64_t current = q.front();
            q.pop();

            for (uint64_t next : graph[current])
            {
                if (d[next] == -1)
                {
                    q.push(next);
                    d[next] = d[current] + 1;

                    // empty the second queue enough to calculate s
                    while (!q2.empty() && q2.front() <= d[next]/2)
                    {
                        q2.pop();
                        s += 1;
                    }
                    if (region.second.find(next) != region.second.end())
                    {
                        q2.push(d[next]);
                    }

                    uint64_t i = next;
                    if (i < p)
                    {
                        if (termsDict.find(i) == termsDict.end())
                            termsDict[i] = std::unordered_map<uint64_t, term_sparse>();
                        if (termsDict[i].find(p) == termsDict[i].end())
                            termsDict[i].insert({ p, term_sparse(i, p, d[next]) });

                        termsDict[i].at(p).w_ij = s / ((double)d[next] * d[next]);
                    }
                    else
                    {
                        if (termsDict.find(p) == termsDict.end())
                            termsDict[p] = std::unordered_map<uint64_t, term_sparse>();
                        if (termsDict[p].find(i) == termsDict[p].end())
                            termsDict[p].insert({ i, term_sparse(p, i, d[next]) });

                        termsDict[p].at(i).w_ji = s / ((double)d[next] * d[next]);
                    }
                }
            }
        }
    }
    // 1-stress
    for (uint64_t i=0; i<n; i++)
    {
        for (uint64_t j : graph[i])
        {
            if (i < j)
            {
                if (termsDict.find(i) == termsDict.end())
                    termsDict[i] = std::unordered_map<uint64_t, term_sparse>();
                if (termsDict[i].find(j) == termsDict[i].end())
                    termsDict[i].insert({ j, term_sparse(i, j, 1) });
                else
                    termsDict[i].at(j).d = 1;

                termsDict[i].at(j).w_ij = termsDict[i].at(j).w_ji = 1;
            }
        }
    }
    std::vector<term_sparse> terms;
    for (const auto& i : termsDict)
    {
        for (const auto& j : i.second)
        {
            terms.push_back(j.second);
        }
    }
    return terms;
}

// returns closest pivot for each vertex, not pivots themselves
std::vector<uint64_t> maxmin_random_sp_weighted(const std::vector<std::vector<edge>>& graph, uint64_t n_pivots, uint64_t p0)
{
    uint64_t n = graph.size();

    std::vector<double> mins(n, std::numeric_limits<double>::max());
    std::vector<uint64_t> argmins(n, -1);

    // first pivot
    mins[p0] = 0;
    argmins[p0] = p0;
    maxmin_bfs_weighted(graph, p0, mins, argmins);
    for (uint64_t i = 0; i < n; i++)
    {
        if (argmins[i] == -1)
            throw "graph has more than one connected component";
    }

    // remaining pivots
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<> rng(0, 1);
    for (uint64_t i = 1; i < n_pivots; i++)
    {
        // choose pivots with probability min
        double min_total = 0;
        for (uint64_t i = 0; i < n; i++)
        {
            min_total += mins[i];
        }
        double rn = rng(mt) * min_total;
        double cumul = 0;
        uint64_t argmax = n-1;
        for (uint64_t i = 1; i < n; i++)
        {
            cumul += mins[i];
            if (cumul >= rn)
            {
                argmax = i;
                break;
            }
        }

        mins[argmax] = 0;
        argmins[argmax] = argmax;
        maxmin_bfs_weighted(graph, argmax, mins, argmins);
    }
    return argmins;
}
void maxmin_bfs_weighted(const std::vector<std::vector<edge>>& graph, const uint64_t p, std::vector<double>& mins, std::vector<uint64_t>& argmins)
{
    uint64_t n = graph.size();
    std::vector<bool> visited(n, false);
    std::vector<double> d(n, std::numeric_limits<double>::max()); // init 'tentative' distances to infinity

    // I am not using a fibonacci heap. I AM NOT USING A FIBONACCI HEAP
    // edges are used 'invisibly' here
    std::priority_queue<edge, std::vector<edge>, edge_comp> pq;

    d[p] = 0;
    pq.push(edge(p,0));

    while (!pq.empty())
    {
        uint64_t current = pq.top().target;
        double d_pi = pq.top().weight;
        pq.pop();
        
        if (!visited[current]) // ignore redundant elements in queue
        {
            visited[current] = true;

            if (d_pi < mins[current])
            {
                mins[current] = d_pi;
                argmins[current] = p;
            }
            for (edge e : graph[current])
            {
                // here the edge is not 'invisible'
                uint64_t next = e.target;
                double weight = e.weight;

                if (d_pi + weight < d[next]) // update tentative value of d 
                {
                    d[next] = d_pi + weight;
                    pq.push(edge(next, d[next]));
                }
            }
        }
    }
}

// again, not a proper MSSP because we get regions for free with maxmin_random_sp
std::vector<term_sparse> MSSP_weighted(const std::vector<std::vector<edge>>& graph, const std::vector<uint64_t>& closest_pivots)
{
    uint64_t n = graph.size();

    // get pivots and their regions, but in sets
    std::unordered_map<uint64_t, std::unordered_set<uint64_t>> regions;
    for (uint64_t i = 0; i < n; i++)
    {
        if (regions.find(closest_pivots[i]) == regions.end())
        {
            regions[closest_pivots[i]] = std::unordered_set<uint64_t>();
        }
        regions[closest_pivots[i]].insert(i);
    }

    std::unordered_map<uint64_t, std::unordered_map<uint64_t, term_sparse>> termsDict;
    for (const auto& region : regions)
    {
        uint64_t p = region.first;

        std::vector<bool> visited(n, false);
        std::vector<double> d(n, std::numeric_limits<double>::max()); // init 'tentative' distances to infinity

        // edges are used 'invisibly' in this queue
        std::priority_queue<edge, std::vector<edge>, edge_comp> pq;

        // init initial edges so that pivot-pivot term is avoided
        for (edge e : graph[p])
        {
            // here the edge is not 'invisible'
            uint64_t next = e.target;
            double weight = e.weight;

            d[next] = weight; // init tentative value of d
            pq.push(edge(next, d[next]));
        }
        d[p] = 0;
        visited[p] = true;

        // q2 contains visited vertices' distances for s calculation
        std::queue<double> q2;
        uint64_t s = 1;

        while (!pq.empty())
        {
            uint64_t current = pq.top().target;
            double d_pi = pq.top().weight;
            pq.pop();
            
            if (!visited[current]) // ignore redundant elements in queue
            {
                visited[current] = true;

                // empty the second queue enough to calculate s
                while (!q2.empty() && q2.front() <= d_pi/2)
                {
                    q2.pop();
                    s += 1;
                }
                if (region.second.find(current) != region.second.end())
                {
                    q2.push(d_pi);
                }

                uint64_t i = current;
                if (i < p)
                {
                    if (termsDict.find(i) == termsDict.end())
                        termsDict[i] = std::unordered_map<uint64_t, term_sparse>();
                    if (termsDict[i].find(p) == termsDict[i].end())
                        termsDict[i].insert({ p, term_sparse(i, p, d_pi) });

                    termsDict[i].at(p).w_ij = s / ((double)d_pi * d_pi);
                }
                else
                {
                    if (termsDict.find(p) == termsDict.end())
                        termsDict[p] = std::unordered_map<uint64_t, term_sparse>();
                    if (termsDict[p].find(i) == termsDict[p].end())
                        termsDict[p].insert({ i, term_sparse(p, i, d_pi) });

                    termsDict[p].at(i).w_ji = s / ((double)d_pi * d_pi);
                }

                // update tentative distances
                for (edge e : graph[current])
                {
                    // here the edge is not 'invisible'
                    uint64_t next = e.target;
                    double weight = e.weight;

                    if (d_pi + weight < d[next]) 
                    {
                        d[next] = d_pi + weight; // update tentative value of d
                        pq.push(edge(next, d[next]));
                    }
                }
            }
        }
    }
    // 1-stress
    for (uint64_t i=0; i<n; i++)
    {
        for (edge e : graph[i])
        {
            uint64_t j = e.target;
            double d_ij = e.weight;
            if (i < j)
            {
                if (termsDict.find(i) == termsDict.end())
                    termsDict[i] = std::unordered_map<uint64_t, term_sparse>();
                if (termsDict[i].find(j) == termsDict[i].end())
                    termsDict[i].insert({ j, term_sparse(i, j, d_ij) });
                else
                    termsDict[i].at(j).d = d_ij;

                termsDict[i].at(j).w_ij = termsDict[i].at(j).w_ji = 1/(d_ij*d_ij);
            }
        }
    }
    std::vector<term_sparse> terms;
    for (const auto& i : termsDict)
    {
        for (const auto& j : i.second)
        {
            terms.push_back(j.second);
        }
    }
    return terms;
}

std::vector<double> schedule(const std::vector<term_sparse> &terms, uint64_t t_max, double eps)
{
    double w_min = std::numeric_limits<double>::max();
    double w_max = std::numeric_limits<double>::min();
    for (const auto& term : terms)
    {
        if (term.w_ij < w_min && term.w_ij != 0) w_min = term.w_ij;
        if (term.w_ji < w_min && term.w_ji != 0) w_min = term.w_ji;

        if (term.w_ij > w_max) w_max = term.w_ij;
        if (term.w_ji > w_max) w_max = term.w_ji;
    }
    double eta_max = 1.0 / w_min;
    double eta_min = eps / w_max;

    double lambda = log(eta_max/eta_min) / ((double)t_max-1);

    // initialize step sizes
    std::vector<double> etas;
    etas.reserve(t_max);
    for (uint64_t t=0; t<t_max; t++)
        etas.push_back(eta_max * exp(-lambda * t));

    return etas;
}


void layout_sparse_unweighted(uint64_t n, double* X, uint64_t m, uint64_t* I, uint64_t* J, uint64_t p, uint64_t t_max, double eps)
{
    try
    {
        std::vector<std::vector<uint64_t>> g = build_graph_unweighted(n, m, I, J);
        auto closest_pivots = maxmin_random_sp_unweighted(g, p, 0);
        auto terms = MSSP_unweighted(g, closest_pivots);
        auto etas = schedule(terms, t_max, eps);
        sgd(X, terms, etas);
    }
    catch (const char* msg)
    {
        std::cerr << "Error: " << msg << std::endl;
    }
}
void layout_sparse_weighted(uint64_t n, double* X, uint64_t m, uint64_t* I, uint64_t* J, double* V, uint64_t p, uint64_t t_max, double eps)
{
    try
    {
        std::vector<std::vector<edge>> g = build_graph_weighted(n, m, I, J, V);

        auto closest_pivots = maxmin_random_sp_weighted(g, p, 0);
        auto terms = MSSP_weighted(g, closest_pivots);
        auto etas = schedule(terms, t_max, eps);
        sgd(X, terms, etas);
    }
    catch (const char* msg)
    {
        std::cerr << "Error: " << msg << std::endl;
    }
}

/*
int main(void)
{
    //uint64_t I[9] = { 0,0,1,2,3,4,4,5,6 };
    //uint64_t J[9] = { 1,2,3,3,4,5,6,7,7 };
    //double V[9] = { 1,1,9,9,1,1,1,1,1 };
    //std::vector<std::vector<edge>> g = build_graph_weighted(8, 9, I, J, V);
        
    try
    {
        uint64_t I[8] = { 0,0,1,2,4,4,5,6 };
        uint64_t J[8] = { 1,2,3,3,5,6,7,7 };
        double V[8] = { 1,1,9,9,1,1,1,1 };
        std::vector<std::vector<edge>> g = build_graph_weighted(8, 8, I, J, V);
        auto closest_pivots = maxmin_random_sp_weighted(g, 2, 0);
    }
    catch (const char* msg)
    {
        std::cerr << "Error: " << msg << std::endl;
    }

    //auto closest_pivots = maxmin_random_sp_weighted(g, 2, 0);
    //for (uint64_t i = 0; i < 8; i++)
    //{
    //    std::cerr << i << " " << closest_pivots[i] << std::endl;
    //}
    //auto terms = MSSP_weighted(g, closest_pivots);
    //for (const auto& term : terms)
    //{
    //    std::cerr << term.i << " " << term.j << " " << term.d << " " << term.w_ij << " " << term.w_ji << std::endl;
    //}
    //auto etas = schedule(terms, 15, .1);
    //for (double eta : etas)
    //{
    //    std::cout << eta << std::endl;
    //}

    //double X[16] = { 0,0, .1,.2, .4,.7, .2,.4, .9,.4, .5,.6, .1,.7, .5,.7 };
    //sgd(X, terms, etas);

    //for (uint64_t i=0; i<8; i++)
    //{
    //    std::cout << X[2 * i] << " " << X[2 * i + 1] << std::endl;
    //}
}
*/


}
