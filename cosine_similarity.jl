using Random
using Distributions
using Statistics


"""
Cosine similarity between vectors v1, v2.
"""
function cos_sim(v1, v2)
    v1dotv2 = sum(v1.*v2)
    v1dotv2 / sqrt(sum(v1.^2) * sum(v2.^2))
end

"""
Generaate a dim dimensional vector with components drawn from uniform
distribution in the range [lower, upper].
"""
function uniform_random_vec(lower, upper, dim)
    return rand(Uniform(lower, upper), dim)
end

"""
Transform a vector so that all components are >= 0, with unit norm.
"""
function exp_transform(v)
    w = exp.(v .- maximum(v))
    w ./ sqrt(sum(w.^2))
end

"""
Create samples number of cosine similarities between pairs of vectors that
have been generated using vector_generator.  Each call to vector_generator()
should generate a single vector.
"""
function sample(vector_generator, samples)
    [cos_sim(vector_generator(), vector_generator()) for _ in 1:samples]
end

"""
Compute (mean, var, min, max) for cosine similarity between vectors generated
using vector_generator.  samples is the number of randome vector pairs that
are generated.
"""
function experiment(vector_generator, samples)
    s = sample(vector_generator, samples)
    (mean(s), var(s), minimum(s), maximum(s))
end

function main(num_samples=10000)
    function uniform(start, stop, dim)
        out = experiment(num_samples) do
            uniform_random_vec(start,stop,dim)
        end
        println("Uniform $start to $stop, d = $dim:")
        println(out)
    end
    function expon(start, stop, dim)
        out = experiment(num_samples) do
            exp_transform(uniform_random_vec(start,stop,dim))
        end
        println("exp_transform, d = $dim:")
        println(out)
    end
    uniform(-1,1,10)
    uniform(-1,1,100)
    uniform(-1,1,1000)
    uniform(0,1,10)
    uniform(0,1,100)
    uniform(0,1,1000)
    expon(-1,1,10)
    expon(-1,1,100)
    expon(-1,1,1000)
end
