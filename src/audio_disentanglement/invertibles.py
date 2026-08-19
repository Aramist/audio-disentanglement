import typing as tp

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

    def log_det_jacobian(self, x: torch.Tensor):
        """Computes the log determinant of the Jacobian of the transformation at x

        Args:
            x (torch.Tensor): The input tensor to be transformed

        Returns:
            torch.Tensor: The log determinant of the Jacobian
        """
        upper_triangular = self.make_upper_triangular()
        # The log determinant of a triangular matrix is the sum of the logs of its diagonal entries
        log_det = torch.sum(torch.log(torch.diagonal(upper_triangular)))
        return log_det

    def inverse(self, x: torch.Tensor):
        """Applies the inverse of the linear transformation to x

        Args:
            x (torch.Tensor): The input tensor to be transformed

        Returns:
            torch.Tensor: The transformed tensor
        """
        raise NotImplementedError("Inverse not implemented yet for LinearTriangular")


class PlanarFlow(Invertible):
    def __init__(
        self, dim: int, nonlinearity: tp.Literal["tanh", "logistic"] = "logistic"
    ):
        """Represents a planar flow transformation
        See: https://arxiv.org/abs/1505.05770

        Args:
            dim (int): The dimensionality of the input and output spaces
        """
        super().__init__()
        self.u = torch.nn.Parameter(torch.empty(dim))
        self.w = torch.nn.Parameter(torch.empty(dim))
        self.b = torch.nn.Parameter(torch.empty(1))
        self.nonlinearity = nonlinearity
        # Initialize the weights randomly
        with torch.no_grad():
            nn.init.normal_(self.u, std=1 / np.sqrt(dim))
            nn.init.normal_(self.w, std=1 / np.sqrt(dim))
            nn.init.zeros_(self.b)

    def _get_parametrized_u(self) -> torch.Tensor | torch.nn.Parameter:
        """Ensures that the transformation is invertible by modifying u if necessary"""
        # Following the derivation for invertibility, since the derivative of the logistic function
        # is bounded by [0, 1/4], it is sufficient to ensure that w^T u > -4
        thresh = -1 if self.nonlinearity == "tanh" else -4
        w_dot_u = torch.dot(self.w, self.u)  # scalar tensor
        m_w_dot_u = thresh + F.softplus(w_dot_u)
        # if w_dot_u <= thresh:
        new_u = self.u + (m_w_dot_u - w_dot_u) * self.w / torch.square(self.w).sum()
        return new_u
        # return self.u

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
        nonlinear_term = (
            torch.tanh(linear_term)
            if self.nonlinearity == "tanh"
            else torch.sigmoid(linear_term)
        )
        return x + self._get_parametrized_u() * nonlinear_term[..., None]

    def inverse(self, x: torch.Tensor):
        """Applies the inverse of the planar flow transformation to x

        Args:
            x (torch.Tensor): The input tensor to be transformed

        Returns:
            torch.Tensor: The transformed tensor
        """
        raise NotImplementedError("Inverse not implemented yet for PlanarFlow")

    def log_det_jacobian(self, x: torch.Tensor):
        """Computes the log determinant of the Jacobian of the transformation at x

        Args:
            x (torch.Tensor): The input tensor to be transformed

        Returns:
            torch.Tensor: The log determinant of the Jacobian
        """
        linear_term = torch.matmul(x, self.w) + self.b  # (*batch,)
        h_prime = 1 - torch.tanh(linear_term) ** 2  # (*batch,)
        psi = h_prime[..., None] * self.w  # (*batch, dim)
        u_dot_psi = torch.matmul(psi, self._get_parametrized_u())  # (*batch,)
        return torch.log(torch.abs(1 + u_dot_psi))  # (*batch,)


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

    def log_det_jacobian(self, x: torch.Tensor):
        """Computes the log determinant of the Jacobian of the transformation at x

        Args:
            x (torch.Tensor): The input tensor to be transformed

        Returns:
            torch.Tensor: The log determinant of the Jacobian
        """
        r = torch.linalg.norm(x - self.z0, dim=-1, keepdim=True)  # (*batch, 1)
        h = 1 / (self.alpha + r)  # (*batch, 1)
        dh_dr = -(h**2)  # (*batch, 1)
        beta = self._get_parametrized_beta()  # (1,)
        d = x.shape[-1]

        return (
            (d - 1) * torch.log(torch.abs(1 + beta * h))
            + torch.log(torch.abs(1 + beta * h + beta * dh_dr * r))
        ).squeeze(-1)


class FullRank(Invertible):
    def __init__(self, dim: int):
        super().__init__()

        self.v = torch.nn.Parameter(torch.empty(dim, dim))
        with torch.no_grad():
            nn.init.kaiming_normal_(self.v)

    def _get_parametrized_v(self) -> torch.Tensor | nn.Parameter:
        """Ensures that the transformation is invertible by modifying v"""
        return self.v.T @ self.v + torch.eye(self.v.shape[0], device=self.v.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self._get_parametrized_v(), None)

    def inverse(self, x: torch.Tensor):
        """Applies the inverse of the radial flow transformation to x

        Args:
            x (torch.Tensor): The input tensor to be transformed

        Returns:
            torch.Tensor: The transformed tensor
        """
        raise NotImplementedError("Inverse not implemented yet for RadialFlow")

    def log_det_jacobian(self, x: torch.Tensor):
        """Computes the log determinant of the Jacobian of the transformation at x

        Args:
            x (torch.Tensor): The input tensor to be transformed

        Returns:
            torch.Tensor: The log determinant of the Jacobian
        """

        batch_shape = x.shape[:-1]
        v_det = torch.linalg.det(
            self.v + torch.eye(self.v.shape[0], device=self.v.device)
        )
        return (2 * torch.log(torch.abs(v_det))).expand(*batch_shape)


class RandPerm(Invertible):
    perm: torch.Tensor
    inv_perm: torch.Tensor

    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer("perm", torch.randperm(dim))
        self.register_buffer("inv_perm", torch.argsort(self.perm))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., self.perm]

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., self.inv_perm]

    def log_det_jacobian(self, x: torch.Tensor):
        # Determinant is always 1
        batch_shape = x.shape[:-1]
        return torch.zeros(batch_shape, device=x.device)


class CouplingFlow(Invertible):
    def __init__(
        self,
        dim: int,
        partition: np.ndarray | int | None = None,
        num_hidden_layers: int = 5,
        hidden_layer_size: int = 1024,
    ):
        super().__init__()
        self.dim = dim
        if isinstance(partition, int):
            self.partition_a_mask = torch.zeros(dim, dtype=torch.bool)
            self.partition_a_mask[:partition] = True
        elif isinstance(partition, np.ndarray):
            self.partition_a_mask = torch.tensor(partition, dtype=torch.bool)
        else:
            self.partition_a_mask = torch.zeros(dim, dtype=torch.bool)
            self.partition_a_mask[: dim // 2] = True

        conditioner_layers = [
            nn.Linear(
                int((~self.partition_a_mask).sum().cpu().item()), hidden_layer_size
            ),
            nn.ReLU(),
        ]
        for _ in range(num_hidden_layers):
            conditioner_layers.append(nn.Linear(hidden_layer_size, hidden_layer_size))
            conditioner_layers.append(nn.ReLU())
        conditioner_layers.append(
            nn.Linear(hidden_layer_size, int(self.partition_a_mask.sum().cpu().item())),
        )
        self.conditioner = nn.Sequential(*conditioner_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_a, x_b = x[..., self.partition_a_mask], x[..., ~self.partition_a_mask]
        s = self.conditioner(x_b)
        x_a = x_a + s

        x = torch.zeros_like(x)
        x[..., self.partition_a_mask] = x_a
        x[..., ~self.partition_a_mask] = x_b
        return x

    def log_det_jacobian(self, x: torch.Tensor) -> torch.Tensor:
        # The Jacobian is lower triangular with ones along the diagonal
        batch_shape = x.shape[:-1]
        return torch.zeros(batch_shape, device=x.device)
