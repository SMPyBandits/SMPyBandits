# Small Julia module to wrap the function that computes a UCB index
module UCBjulia
    function index(rewards, pulls, t, arm, alpha=4)
        if pulls[arm] < 1
            return Inf
        else
            return (rewards[arm] / pulls[arm]) + sqrt((alpha * log(t)) / (2 * pulls[arm]))
        end
    end
end

# Small Julia function that computes a UCB index
function index(rewards, pulls, t, arm, alpha=4)
    if pulls[arm] < 1
        return Inf
    else
        return (rewards[arm] / pulls[arm]) + sqrt((alpha * log(t)) / (2 * pulls[arm]))
    end
end