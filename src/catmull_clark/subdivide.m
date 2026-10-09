% subdivide.m - Perform one subdivision step on a triangular mesh
%
%   [Vnew, Fnew] = subdivide(V, F) takes an input mesh defined by
%   vertices V (Nx3) and triangular faces F (Mx3) and returns a refined
%   mesh. The implementation follows a simple scheme where each triangle
%   is split into four smaller triangles by adding the face centroid as a
%   new vertex.
%
%   Input:
%       V - Nx3 array of vertex coordinates.
%       F - Mx3 array of indices into V defining triangular faces.
%   Output:
%       Vnew - (N+M)x3 array of refined vertex coordinates.
%       Fnew - (3*M)x3 array of refined triangular faces.
%
%   This function is deliberately simple and serves as a foundation for
%   more sophisticated Catmull‑Clark or Loop subdivision schemes.
%
%   Example:
%       V = [0 0 0; 1 0 0; 0 1 0];
%       F = [1 2 3];
%       [Vnew, Fnew] = subdivide(V, F);
%
%   See also: basisFunctions.

function [Vnew, Fnew] = subdivide(V, F)
    % Validate inputs
    arguments
        V (:,3) double
        F (:,3) double {mustBePositive, mustBeInteger}
    end

    % Number of original vertices and faces
    numV = size(V, 1);
    numF = size(F, 1);

    % Compute face centroids (new vertices)
    faceCentroids = zeros(numF, 3);
    for f = 1:numF
        verts = V(F(f, :), :);
        faceCentroids(f, :) = mean(verts, 1);
    end

    % Append new vertices to the vertex list
    Vnew = [V; faceCentroids];

    % Build new faces: each original triangle (i,j,k) becomes three
    % triangles (i,j,c), (j,k,c), (k,i,c) where c is the index of the
    % centroid vertex.
    centroidOffset = numV; % first centroid index in Vnew
    Fnew = zeros(numF * 3, 3);
    idx = 1;
    for f = 1:numF
        i = F(f, 1);
        j = F(f, 2);
        k = F(f, 3);
        c = centroidOffset + f; % centroid vertex index
        Fnew(idx, :) = [i, j, c]; idx = idx + 1;
        Fnew(idx, :) = [j, k, c]; idx = idx + 1;
        Fnew(idx, :) = [k, i, c]; idx = idx + 1;
    end
end
