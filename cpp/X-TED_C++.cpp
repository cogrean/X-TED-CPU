#include <iostream>
#include <vector>
#include <time.h>
#include <string>
#include "fstream"
#include "thread"
#include "queue"
#include "unordered_set"
#include "set"
#include <chrono>
#include <functional>

namespace xted
{

    using std::string;
    using std::thread;
    using std::vector;

    namespace
    {
        /*
        Params:
            adj: pointer to 2D vector where adj[i] == {1D vector of children of tree[i]}
        Returns: vector containing the right-most leaf possible of accessing from each node[i]
        */
        vector<int> right_leaf_preprocessing(vector<vector<int>> &adj)
        {
            int m = (int)adj.size();
            vector<int> orl(m, -1);
            int r;
            int i;

            for (i = 0; i < m; i++)
            {
                r = i;
                while (true)
                {
                    if ((int)adj[r].size() == 0)
                    {
                        orl[i] = r;
                        break;
                    }
                    if (orl[r] >= 0)
                    {
                        orl[i] = orl[r];
                        break;
                    }
                    r = adj[r][adj[r].size() - 1];
                }
            }
            return orl;
        }
        /*
        calculates key roots of a tree given the rightmost-leaf array
         */
        vector<int> key_roots(vector<int> &orl)
        {
            int m = (int)orl.size();
            vector<int> kr_view(m, -1);
            int K = 0;
            int i;
            int r;
            for (i = 0; i < m; i++)
            {
                r = orl[i];
                if (kr_view[r] < 0)
                {
                    kr_view[r] = i;
                    K += 1;
                }
            }
            vector<int> key_roots_view(K, 0);
            int k;
            int j;
            i = 0;
            for (k = 0; k < K; k++)
            {
                while (kr_view[i] < 0)
                {
                    i += 1;
                }
                j = k;
                while ((j > 0) && (key_roots_view[j - 1] < kr_view[i]))
                {
                    key_roots_view[j] = key_roots_view[j - 1];
                    j -= 1;
                }

                key_roots_view[j] = kr_view[i];
                i += 1;
            }
            return key_roots_view;
        }

        //branchless compilations (cmov) for x86, and standard implementation (2 comparisons) for non-x86
        inline int min3(int a, int b, int c)
        {
            int ab = a < b ? a : b;
            return ab < c ? ab : c;
        }

    }

    // forward declarations
    void compute(int k, int l, vector<int> &x_orl, vector<int> &x_kr, vector<int> &y_orl, vector<int> &y_kr, vector<vector<int>> &Cost, vector<vector<int>> &D, vector<vector<int>> &D_tree);
    vector<vector<int>> parallel_CPU_compute(vector<int> &x_orl, vector<int> &x_kr, vector<int> &y_orl, vector<int> &y_kr, vector<vector<int>> &Cost, vector<vector<int>> &D_tree, int m, int n, int num_threads);

    // thread task
    void task(vector<int> &worklist_1, int begin, int interval, int final, int L, vector<int> &x_orl, vector<int> &x_kr, vector<int> &y_orl, vector<int> &y_kr, vector<vector<int>> &Cost, vector<vector<int>> &D, vector<vector<int>> &D_tree)
    {
        int i = begin;
        while (i < final)
        {
            int task = worklist_1[i];
            worklist_1[i] = -1;

            // computes one table (inlined the compute_one_table function call in original code to reduce overhead)
            int row = task / L;
            int column = task % L;
            compute(row, column, x_orl, x_kr, y_orl, y_kr, Cost, D, D_tree);

            //advances
            i = i + interval;
        }
    }

    //computes table
    void compute(int k, int l, vector<int> &x_orl, vector<int> &x_kr, vector<int> &y_orl, vector<int> &y_kr, vector<vector<int>> &Cost, vector<vector<int>> &D, vector<vector<int>> &D_tree)
    {
        int i;
        int j;

        int i_0;
        int j_0;
        int i_max;
        int j_max;

        i_0 = x_kr[k];
        j_0 = y_kr[l];
        i_max = x_orl[i_0] + 1;
        j_max = y_orl[j_0] + 1;
        D[i_max][j_max] = 0;

        for (i = i_max - 1; i > i_0 - 1; i--)
        {
            D[i][j_max] = 1 + D[i + 1][j_max];
        }

        for (j = j_max - 1; j > j_0 - 1; j--)
        {
            D[i_max][j] = 1 + D[i_max][j + 1];
        }

        for (i = i_max - 1; i > i_0 - 1; i--)
        {
            for (j = j_max - 1; j > j_0 - 1; j--)
            {

                if ((x_orl[i] == x_orl[i_0]) && (y_orl[j] == y_orl[j_0]))
                {

                    D[i][j] = min3(Cost[i][j] + D[i + 1][j + 1], 1 + D[i + 1][j], 1 + D[i][j + 1]);
                    D_tree[i][j] = D[i][j];
                }
                else
                {

                    D[i][j] = min3(D_tree[i][j] + D[x_orl[i] + 1][y_orl[j] + 1], 1 + D[i + 1][j],
                                   1 + D[i][j + 1]);
                }
            }
        }
    }

    /*
    entrypoint for x_ted_compute from Dayi's proposal. Prepocesses and prepares trees for parallel XTED_CPU computation
    Returns: returns the final distance
    */
    int XTED_CPU(vector<string> label1, vector<vector<int>> parent1, vector<string> label2, vector<vector<int>> parent2, vector<vector<int>> cost_matrix, int num_threads)
    {

        // outer-most leaf arrays
        vector<int> x_orl = right_leaf_preprocessing(ref(parent1));
        vector<int> y_orl = right_leaf_preprocessing(ref(parent2));

        // keyroot tree arrays
        vector<int> x_kr = key_roots(ref(x_orl));
        vector<int> y_kr = key_roots(ref(y_orl));

        int m = (int)label1.size();
        int n = (int)label2.size();

        // D_tree: per-cell tree-to-tree distances, shared across threads (non-overlapping jobs)
        vector<vector<int>> D_tree(m, vector<int>(n, -1));

        vector<vector<int>> result_matrix = parallel_CPU_compute(x_orl, x_kr, y_orl, y_kr, cost_matrix, D_tree, m, n, num_threads);

        return result_matrix[0][0];
    }

    /*
    Uniform-cost variant: rename costs 0 for matching labels, 1 otherwise. Builds the cost
    matrix internally so no Python-side construction or pybind11 STL conversion is needed.
    */
    int XTED_CPU_uniform(vector<string> label1, vector<vector<int>> parent1, vector<string> label2, vector<vector<int>> parent2, int num_threads)
    {
        vector<int> x_orl = right_leaf_preprocessing(ref(parent1));
        vector<int> y_orl = right_leaf_preprocessing(ref(parent2));

        vector<int> x_kr = key_roots(ref(x_orl));
        vector<int> y_kr = key_roots(ref(y_orl));

        int m = (int)label1.size();
        int n = (int)label2.size();

        vector<vector<int>> cost_matrix(m, vector<int>(n));
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j)
                cost_matrix[i][j] = (label1[i] == label2[j]) ? 0 : 1;

        vector<vector<int>> D_tree(m, vector<int>(n, -1));

        vector<vector<int>> result_matrix = parallel_CPU_compute(x_orl, x_kr, y_orl, y_kr, cost_matrix, D_tree, m, n, num_threads);

        return result_matrix[0][0];
    }

    /*
    Parameters:
        m & n: size of tree X and Y
        x_orl & y_orl: arrays with size m and n containing each node (X[i]'s/Y[i]'s) right-most leaf node
        x_kr & y_kr: arrays containing the keyroots of X and Y
        Cost: 2D integer matrix of size m * n of node rename costs
        D_tree: Final Tree-to-Tree edit distances. Shared between threads due to non-overlapping jobs. contains final TED value at D_tree[0][0]
        num_threads: thread count for computation
    Returns:
        Full cost array with final TED value at D_tree[0][0]
    */
    vector<vector<int>> parallel_CPU_compute(vector<int> &x_orl, vector<int> &x_kr, vector<int> &y_orl, vector<int> &y_kr, vector<vector<int>> &Cost, vector<vector<int>> &D_tree, int m, int n, int num_threads)
    {
        int K = (int)x_kr.size();
        int L = (int)y_kr.size();

        //creates num_th amount of D trees for threads to work within
        int num_th = num_threads;
        vector<vector<vector<int>>> D_in_total;
        for (int i = 0; i < num_th; i++)
        {
            vector<vector<int>> D_new(m + 1, vector<int>(n + 1, -1));
            D_in_total.push_back(D_new);
        }

        vector<int> depth(K * L, 0);

        vector<int> x_keyroot_depth(K, 0);
        vector<int> y_keyroot_depth(L, 0);

        // Preprocessing begins

        //assigns depth to each keyroot for X
        for (int i = 0; i < K; i++)
        {
            int node = x_kr[i];
            if ((node == x_orl[node]) || (node == x_orl[node] - 1))
            {
                x_keyroot_depth[i] = 0;
            }
            else
            {
                for (int j = 0; j < i; j++)
                {
                    int node_2 = x_kr[j];
                    if ((node <= node_2) && (x_orl[node] >= node_2))
                    {
                        x_keyroot_depth[i] = std::max(x_keyroot_depth[i], x_keyroot_depth[j] + 1);
                    }
                }
            }
        }

        //depth assignment for Y
        for (int i = 0; i < L; i++)
        {
            int node = y_kr[i];
            if ((node == y_orl[node]) || (node == y_orl[node] - 1))
            {
                y_keyroot_depth[i] = 0;
            }
            else
            {
                for (int j = 0; j < i; j++)
                {
                    int node_2 = y_kr[j];
                    if ((node <= node_2) && (y_orl[node] >= node_2))
                    {
                        y_keyroot_depth[i] = std::max(y_keyroot_depth[i], y_keyroot_depth[j] + 1);
                    }
                }
            }
        }

        //depth calculation considering both trees
        for (int i = 0; i < K * L; i++)
        {
            depth[i] = x_keyroot_depth[i / L] + y_keyroot_depth[i % L];
        }

        // Preprocessing ends

        //TODO: add logging for time it takes to preprocess and compare against running times

        vector<int> worklist1(K * L, -1);
        vector<int> worklist2(K * L, -1);
        int worklist1_tail = 0;
        int worklist2_tail = 0;

        //first filling of the worklist vector 
        int current_depth = 0;
        for (int i = 0; i < K * L; i++)
        {
            if (depth[i] == current_depth)
            {
                worklist1[worklist1_tail++] = i;
            }
        }

        int max = 0;
        for (int i = 0; i < (int)depth.size(); i++)
        {
            if (max < depth[i])
            {
                max = depth[i];
            }
        }

        double total_time = 0;

        while (worklist1_tail != 0)
        {
            auto start_time = std::chrono::steady_clock::now();

            vector<thread> threads;

            for (int inter = 0; inter < num_th; inter++)
            {
                threads.push_back(thread(task, ref(worklist1), inter, num_th, (int)worklist1_tail, L, ref(x_orl), ref(x_kr), ref(y_orl), ref(y_kr), ref(Cost), ref(D_in_total[inter]), ref(D_tree)));
            }

            for (auto &th : threads)
            {
                th.join();
            }

            auto end_time = std::chrono::steady_clock::now();
            auto ms = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
            total_time += static_cast<double>(ms / 1000.0);

            //update current depth and refill worklist
            current_depth++;
            for (int i = 0; i < K * L; i++)
            {
                if (depth[i] == current_depth)
                {
                    worklist2[worklist2_tail++] = i;
                }
            }

            swap(worklist1, worklist2);
            worklist1_tail = worklist2_tail;
            worklist2_tail = 0;
        }

        std::cout << "Total Time for Parallel Computing = " << total_time << " ms" << std::endl;
        vector<vector<int>> final_result = D_tree;
        return final_result;
    }

}
