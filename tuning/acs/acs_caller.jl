#!/usr/bin/env julia
include("../../scripts/acs.jl")

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
        "--m-colony"
        arg_type = Int
        required = true
        "--local-phero-decay"
        arg_type = Float64
        required = true
        "--phero-decay"
        arg_type = Float64
        required = true
        "--beta"
        arg_type = Float64
        required = true
        "--alpha"
        arg_type = Float64
        required = true
        "--factor-greed"
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
    (_, _, _, best_score) = ant_colony_system(
        instance;
        solution=init_solution,
        score=init_score,
        n_iterations=args["n-iterations"],
        m_colony=args["m-colony"],
        local_phero_decay=args["local-phero-decay"],
        phero_decay=args["phero-decay"],
        beta=args["beta"],
        alpha=args["alpha"],
        factor_greed=args["factor-greed"],
        verbose=false,
    )
    end_time = time()
    run_time = max(end_time - start_time, 0.0000001)
    println("$best_score $run_time")
end

main()