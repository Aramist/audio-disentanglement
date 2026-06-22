import torch
from torch import nn


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
