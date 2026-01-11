using Plots, Printf, Logging
include("instance_parsing.jl")

function log_unsatisfied(served_requests::Int64, instance_γ::Int64, verbose::Bool=true)
    if verbose
        @warn "Not enough requests served - $served_requests, out of $instance_γ"
    end
end

function log_unfeasable(verbose::Bool=true)
    if verbose
        @error "Did not find feasible solution!"
    end
end

function log_iteration(iter_n::Int64, best_score::Float64, verbose::Bool=true)
    if verbose
        @info "Iteration $(iter_n+1) yielded best score: $(@sprintf("%.2f", best_score))."
    end
end

function log_result(instance::PDPInstance, best_solution::PDPSolutionVector, best_score::Float64, verbose::Bool=true)
    if verbose
        @info "Found $(is_feasible(instance, best_solution) ? "feasible" : "infeasible") solution with quality $best_score."
    end
end