struct Permutation{N}
    position::Int

    function Permutation{N}(position) where {N}
        if position < 0 || position > prod(1:N)
            throw(BoundsError(1:prod(1:N), position))
        end
        new{N}(position)
    end
end

Base.zero(::Type{Permutation{N}}) where {N} = Permutation{N}(1)
Base.convert(::Type{Int}, p::Permutation{N}) where {N} = p.position

_permcomposition(::Val{1}) = SA{Int8}[1]
_perminversion(::Val{1}) = SA{Int8}[1]

_permcomposition(::Val{2}) = SA{Int8}[1 2; 2 1]
_perminversion(::Val{2}) = SA{Int8}[1, 2]

_permcomposition(::Val{3}) = SA{Int8}[1  2  3  4  5  6
                                      2  1  5  6  3  4
                                      3  4  1  2  6  5
                                      4  3  6  5  1  2
                                      5  6  2  1  4  3
                                      6  5  4  3  2  1]
_perminversion(::Val{3}) = SA{Int8}[1, 2, 3, 5, 4, 6]

_permcomposition(::Val{4}) = SA{Int8}[ 1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24;
                                       2   1   5   6   3   4   8   7  11  12   9  10  19  20  21  22  23  24  13  14  15  16  17  18;
                                       3   4   1   2   6   5  13  14  15  16  17  18   7   8   9  10  11  12  20  19  23  24  21  22;
                                       4   3   6   5   1   2  14  13  17  18  15  16  20  19  23  24  21  22   7   8   9  10  11  12;
                                       5   6   2   1   4   3  19  20  21  22  23  24   8   7  11  12   9  10  14  13  17  18  15  16;
                                       6   5   4   3   2   1  20  19  23  24  21  22  14  13  17  18  15  16   8   7  11  12   9  10;
                                       7   8   9  10  11  12   1   2   3   4   5   6  15  16  13  14  18  17  21  22  19  20  24  23;
                                       8   7  11  12   9  10   2   1   5   6   3   4  21  22  19  20  24  23  15  16  13  14  18  17;
                                       9  10   7   8  12  11  15  16  13  14  18  17   1   2   3   4   5   6  22  21  24  23  19  20;
                                      10   9  12  11   7   8  16  15  18  17  13  14  22  21  24  23  19  20   1   2   3   4   5   6;
                                      11  12   8   7  10   9  21  22  19  20  24  23   2   1   5   6   3   4  16  15  18  17  13  14;
                                      12  11  10   9   8   7  22  21  24  23  19  20  16  15  18  17  13  14   2   1   5   6   3   4;
                                      13  14  15  16  17  18   3   4   1   2   6   5   9  10   7   8  12  11  23  24  20  19  22  21;
                                      14  13  17  18  15  16   4   3   6   5   1   2  23  24  20  19  22  21   9  10   7   8  12  11;
                                      15  16  13  14  18  17   9  10   7   8  12  11   3   4   1   2   6   5  24  23  22  21  20  19;
                                      16  15  18  17  13  14  10   9  12  11   7   8  24  23  22  21  20  19   3   4   1   2   6   5;
                                      17  18  14  13  16  15  23  24  20  19  22  21   4   3   6   5   1   2  10   9  12  11   7   8;
                                      18  17  16  15  14  13  24  23  22  21  20  19  10   9  12  11   7   8   4   3   6   5   1   2;
                                      19  20  21  22  23  24   5   6   2   1   4   3  11  12   8   7  10   9  17  18  14  13  16  15;
                                      20  19  23  24  21  22   6   5   4   3   2   1  17  18  14  13  16  15  11  12   8   7  10   9;
                                      21  22  19  20  24  23  11  12   8   7  10   9   5   6   2   1   4   3  18  17  16  15  14  13;
                                      22  21  24  23  19  20  12  11  10   9   8   7  18  17  16  15  14  13   5   6   2   1   4   3;
                                      23  24  20  19  22  21  17  18  14  13  16  15   6   5   4   3   2   1  12  11  10   9   8   7;
                                      24  23  22  21  20  19  18  17  16  15  14  13  12  11  10   9   8   7   6   5   4   3   2   1]
_perminversion(::Val{4}) = SA{Int8}[1, 2, 3, 5, 4, 6, 7, 8, 13, 19, 14, 20, 9, 11, 15, 21, 17, 23, 10, 12, 16, 22, 18, 24]

function Base.inv(p::Permutation{N}) where {N}
    return @inbounds Permutation{N}(_perminversion(Val(N))[p.position])
end

function Base.:(∘)(p::Permutation{N}, q::Permutation{N}) where {N}
    return @inbounds Permutation{N}(_permcomposition(Val(N))[p.position, q.position])
end

Base.@propagate_inbounds function getpermutedindex(A, p::Permutation{1},
                                                   i::Integer)
    return A[i]
end

Base.@propagate_inbounds function getpermutedindex(A, p::Permutation{2}, i)
    ax = p == Permutation{2}(1) ? axes(A) : reverse.(axes(A))
    return A[CartesianIndices(ax)[i]]
end

Base.@propagate_inbounds function getpermutedindex(A, p::Permutation{4}, i)
    if p == Permutation{4}(1)
        #   4---5---6     4---5---6
        #   |       | --> |       |
        #   1---2---3     1---2---3
        return A[i]
    elseif p == Permutation{4}(3)
        #   4---5---6     3---6
        #   |       | --> |   |
        #   1---2---3     2   5
        #                 |   |
        #                 1---4
        a, b = axes(A)
        j = CartesianIndices((b,a))[i]
        return A'[j]
    elseif p == Permutation{4}(8)
        #   4---5---6     6---5---4
        #   |       | --> |       |
        #   1---2---3     3---2---1
        a, b = axes(A)
        return A[CartesianIndices((reverse(a),b))[i]]
    elseif p == Permutation{4}(11)
        #   4---5---6     1---4
        #   |       | --> |   |
        #   1---2---3     2   5
        #                 |   |
        #                 3---6
        a, b = axes(A)
        j = CartesianIndices((b,reverse(a)))[i]
        return A'[j]
    elseif p == Permutation{4}(14)
        #   4---5---6     6---3
        #   |       | --> |   |
        #   1---2---3     5   2
        #                 |   |
        #                 4---1
        a, b = axes(A)
        j = CartesianIndices((reverse(b),a))[i]
        return A'[j]
    elseif p == Permutation{4}(17)
        #   4---5---6     1---2---3
        #   |       | --> |       |
        #   1---2---3     4---5---6
        a, b = axes(A)
        j = CartesianIndices((a,reverse(b)))[i]
        return A[j]
    elseif p == Permutation{4}(22)
        #   4---5---6     4---1
        #   |       | --> |   |
        #   1---2---3     5   2
        #                 |   |
        #                 6---3
        a, b = axes(A)
        j = CartesianIndices((reverse(b),reverse(a)))[i]
        return A'[j]
    elseif p == Permutation{4}(24)
        #   4---5---6     3---2---1
        #   |       | --> |       |
        #   1---2---3     6---5---4
        a, b = axes(A)
        j = CartesianIndices((reverse(a),reverse(b)))[i]
        return A[j]
    else
        throw(BoundsError())
    end
end
