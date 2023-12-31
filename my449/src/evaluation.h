#ifndef EVALUATION_H
#define EVALUATION_H

#include "expression.h"
#include <unordered_map>
#include <vector>

class evaluation
{
public:
    evaluation(const std::vector<expression> &exprs);

    void add_kwargs_double(
        const char *key,
        double value);

    void add_kwargs_ndarray(
        const char *key,
        int dim,
        size_t shape[],
        double data[]);

    // return 0 for success
    int execute();

    // return the variable computed by the last expression
    double &get_result();

private:
    double result_;
    std::unordered_map<std::string, double> scalars_;
    std::unordered_map<std::string, std::pair<std::vector<size_t>, std::vector<double>>> tensors_;
    std::vector<expression> expressions_;
}; // class evaluation

#endif // EVALUATION_H
