% basisFunctions.m - Evaluate linear basis (barycentric) on a triangular mesh
%
%   B = basisFunctions(V, F, P) returns a matrix of basis function values
%   evaluated at a set of query points P. Each row of B corresponds to a
%   point in P and contains the barycentric coordinates with respect to the
%   triangle that contains the point. The columns correspond to the mesh
%   vertices (size(V,1)). Points that lie outside the mesh produce a row of
%   zeros.
%
%   Input:
%       V - Nx3 array of vertex coordinates.
%       F - Mx3 array of triangular face indices (1‑based).
%       P - Kx3 array of query point coordinates.
%   Output:
%       B - KxN matrix of basis function values.
%
%   This implementation is intentionally simple and not optimised for
%   performance. It iterates over each point and each triangle until it
%   finds a containing triangle, then computes barycentric coordinates.
%   For large meshes a spatial acceleration structure would be preferred.
%
%   See also: subdivide.

function B = basisFunctions(V, F, P)
    arguments
        V (:,3) double
        F (:,3) double {mustBePositive, mustBeInteger}
        P (:,3) double
    end

    numV = size(V,1);
    numP = size(P,1);
    B = zeros(numP, numV);

    % Pre‑compute edge vectors for each triangle
    triVerts1 = V(F(:,1),:);
    triVerts2 = V(F(:,2),:);
    triVerts3 = V(F(:,3),:);
    e0 = triVerts2 - triVerts1; % v1 - v0
    e1 = triVerts3 - triVerts1; % v2 - v0
    denom = cross(e0, e1, 2);
    area2 = sqrt(sum(denom.^2,2)); % 2 * triangle area

    for pIdx = 1:numP
        p = P(pIdx,:);
        found = false;
        for fIdx = 1:size(F,1)
            v0 = triVerts1(fIdx,:);
            v1 = triVerts2(fIdx,:);
            v2 = triVerts3(fIdx,:);
            % Compute barycentric coordinates using vector formulation
            v0v1 = v1 - v0;
            v0v2 = v2 - v0;
            v0p  = p  - v0;
            d00 = dot(v0v1, v0v1);
            d01 = dot(v0v1, v0v2);
            d11 = dot(v0v2, v0v2);
            d20 = dot(v0p , v0v1);
            d21 = dot(v0p , v0v2);
            denomB = d00*d11 - d01*d01;
            if denomB == 0
                continue;
            end
            v = (d11*d20 - d01*d21) / denomB;
            w = (d00*d21 - d01*d20) / denomB;
            u = 1 - v - w;
            % Check if point lies inside triangle (allow small tolerance)
            if u >= -1e-9 && v >= -1e-9 && w >= -1e-9
                % Assign barycentric weights to the three vertices
                B(pIdx, F(fIdx,1)) = u;
                B(pIdx, F(fIdx,2)) = v;
                B(pIdx, F(fIdx,3)) = w;
                found = true;
                break;
            end
        end
        % If not found, row remains zeros.
    end
end
