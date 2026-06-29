import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Invertible(nn.Module):
    def __init__(self):
        """Represents a homeomorphism on some real vector space"""
        super().__init__()

    def forward(self, x: torch.Tensor):
        """Applies an invertible transformation to x

        Args:
            x (torch.Tensor): The input tensor to be transformed

        Raises:
            NotImplementedError: This method should be implemented by subclasses of Invertable
        """
        raise NotImplementedError()

    def inverse(self, x: torch.Tensor):
        """Applies the inverse of the invertible transformation to x

        Args:
            x (torch.Tensor): The input tensor to be transformed

        Raises:
            NotImplementedError: This method should be implemented by subclasses of Invertable
        """
        raise NotImplementedError()


class Identity(Invertible):
    def forward(self, x: torch.Tensor):
        """Returns the input tensor unchanged

        Args:
            x (torch.Tensor): The input tensor to be transformed

        Returns:
            torch.Tensor: The input tensor unchanged
        """
        return x

    def inverse(self, x: torch.Tensor):
        """Returns the input tensor unchanged

        Args:
            x (torch.Tensor): The input tensor to be transformed

        Returns:
            torch.Tensor: The input tensor unchanged
        """
        return x


class LinearTriangular(Invertible):
    def __init__(self, dim: int):
        """Represents an invertible linear transformation parametrized by an upper
        triangular matrix with non-zero diagonal entries.

        Args:
            dim (int): The dimensionality of the input and output spaces
        """
        super().__init__()
        self.proj = torch.nn.Parameter(torch.empty((dim, dim)))
        self.bias = torch.nn.Parameter(torch.empty(dim))
        # Initialize the weights randomly
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.proj)
            nn.init.zeros_(self.bias)

    def make_upper_triangular(self):
        """Constructs an upper-triangular matrix with non-zero main diagonal elements
        parametrized by the stored projection matrix
        """
        upper_triangle = torch.triu(self.proj)
        # Ensure the diagonal entries are non-zero by applying softplus
        diag_indices = torch.arange(self.proj.shape[0]).to(self.proj.device)
        upper_triangle[diag_indices, diag_indices] = torch.nn.functional.softplus(
            upper_triangle[diag_indices, diag_indices]
        )
        return upper_triangle

    def forward(self, x: torch.Tensor):
        """Applies the linear transformation to x

        Args:
            x (torch.Tensor): The input tensor to be transformed

        Returns:
            torch.Tensor: The transformed tensor
        """
        return torch.nn.functional.linear(x, self.make_upper_triangular(), self.bias)

    def inverse(self, x: torch.Tensor):
        """Applies the inverse of the linear transformation to x

        Args:
            x (torch.Tensor): The input tensor to be transformed

        Returns:
            torch.Tensor: The transformed tensor
        """
        raise NotImplementedError("Inverse not implemented yet for LinearTriangular")


class PlanarFlow(Invertible):
    def __init__(self, dim: int):
        """Represents a planar flow transformation
        See: https://arxiv.org/abs/1505.05770

        Args:
            dim (int): The dimensionality of the input and output spaces
        """
        super().__init__()
        self.u = torch.nn.Parameter(torch.empty(dim))
        self.w = torch.nn.Parameter(torch.empty(dim))
        self.b = torch.nn.Parameter(torch.empty(1))
        # Initialize the weights randomly
        with torch.no_grad():
            nn.init.normal_(self.u, std=1 / np.sqrt(dim))
            nn.init.normal_(self.w, std=1 / np.sqrt(dim))
            nn.init.zeros_(self.b)

    def _get_parametrized_u(self) -> torch.Tensor | torch.nn.Parameter:
        """Ensures that the transformation is invertible by modifying u if necessary"""
        w_dot_u = torch.dot(self.w, self.u)  # scalar tensor
        m_w_dot_u = -1 + F.softplus(w_dot_u)
        if w_dot_u < -1:
            new_u = self.u + (m_w_dot_u - w_dot_u) * self.w / torch.square(self.w).sum()
            return new_u
        return self.u

    def forward(self, x: torch.Tensor):
        """Applies the planar flow transformation to x

        Args:
            x (torch.Tensor): The input tensor to be transformed  (*batch, dim)

        Returns:
            torch.Tensor: The transformed tensor
        """
        linear_term = torch.matmul(x, self.w) + self.b  # (*batch,)
        # In theory, h can be any smooth, non-linear function. Here we use
        # tanh as this is used in the original paper.
        return x + self._get_parametrized_u() * torch.tanh(linear_term)[..., None]

    def inverse(self, x: torch.Tensor):
        """Applies the inverse of the planar flow transformation to x

        Args:
            x (torch.Tensor): The input tensor to be transformed

        Returns:
            torch.Tensor: The transformed tensor
        """
        raise NotImplementedError("Inverse not implemented yet for PlanarFlow")


class RadialFlow(Invertible):
    def __init__(self, dim: int):
        """Represents a radial flow transformation
        See: https://arxiv.org/abs/1505.05770

        Args:
            dim (int): The dimensionality of the input and output spaces
        """
        super().__init__()
        self.z0 = torch.nn.Parameter(torch.empty(dim))
        self.alpha = torch.nn.Parameter(torch.empty(1))
        self.beta = torch.nn.Parameter(torch.empty(1))
        # Initialize the weights randomly
        with torch.no_grad():
            nn.init.normal_(self.z0, std=1 / np.sqrt(dim))
            nn.init.normal_(self.alpha)
            nn.init.normal_(self.beta)

    def _get_parametrized_beta(self) -> torch.Tensor | torch.nn.Parameter:
        """Ensures that the transformation is invertible by modifying beta if necessary"""
        return -self.alpha + F.softplus(self.beta)

    def forward(self, x: torch.Tensor):
        """Applies the radial flow transformation to x

        Args:
            x (torch.Tensor): The input tensor to be transformed

        Returns:
            torch.Tensor: The transformed tensor
        """
        r = torch.norm(x - self.z0, dim=-1, keepdim=True)  # (*batch, 1)
        h = 1 / (self.alpha + r)  # (*batch, 1)
        return x + self._get_parametrized_beta() * h * (x - self.z0)

    def inverse(self, x: torch.Tensor):
        """Applies the inverse of the radial flow transformation to x

        Args:
            x (torch.Tensor): The input tensor to be transformed

        Returns:
            torch.Tensor: The transformed tensor
        """
        raise NotImplementedError("Inverse not implemented yet for RadialFlow")
