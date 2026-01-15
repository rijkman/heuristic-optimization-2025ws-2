#!/usr/bin/env julia
include("../../scripts/lns.jl")

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--seed"
        arg_type = Int64
        required = true
        "--instance"
        arg_type = String
        required = true
        "--n-iterations"
        arg_type = Int
        required = true
        "--m-repairs"
        arg_type = Int
        required = true
        "--alpha"
        arg_type = Float64
        required = true
    end

    return parse_args(s)
end

function main()
    args = parse_commandline()
    Random.seed!(args["seed"])

    instance_data = args["instance"]
    instance = read_PDPInstance(instance_data)
    (_, _, init_solution, init_score) = greedy_heuristic_one_extend_random(instance, verbose=false)

    start_time = time()
    (_, _, _, best_score) = large_neighborhood_search(
        instance;
        solution=init_solution,
        score=init_score,
        n_iterations=args["n-iterations"],
        m_repairs=args["m-repairs"],
        alpha=args["alpha"],
        verbose=false,
    )
    end_time = time()
    run_time = max(end_time - start_time, 0.0000001)
    println("$best_score $run_time")
end

main()