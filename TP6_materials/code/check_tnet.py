import torch
import numpy as np
import matplotlib.pyplot as plt
from pointnet import Tnet


def generate_highly_asymmetric_t_structure(n_points=1024):
    """
    Generates a synthetic point cloud with a highly asymmetric T-like structure,
    having easy directions to spot.
    n_points: int, total number of points in the point cloud

    Returns:
    - points: torch.Tensor, the generated point cloud with shape (n_points, 3)
    """
    # base of the T
    base_x = np.random.uniform(-2, 1, n_points // 2)
    base_y = np.random.uniform(-0.1, 0.2, n_points // 2)
    base_z = np.random.uniform(-0.3, 0.5, n_points // 2)
    # vertical part of the T
    vertical_x = np.random.uniform(-0.2, 0.2, n_points // 4)
    vertical_y = np.random.uniform(-2, 2, n_points // 4)
    vertical_z = np.random.uniform(-0.2, 0.2, n_points // 4)
    # additional points
    additional_x = np.random.uniform(0.5, 1.5, n_points // 4)
    additional_y = np.random.uniform(-1, 1, n_points // 4)
    additional_z = np.random.uniform(-0.5, 0.8, n_points // 4)

    # Combine the parts
    x = np.concatenate([base_x, 1+vertical_x, -5+additional_x])
    y = np.concatenate([base_y, vertical_y, -2+additional_y])
    z = np.concatenate([base_z, vertical_z, 1+additional_z])

    points = np.vstack((x, y, z)).T
    points = torch.tensor(points, dtype=torch.float32)
    points -= points.mean(0)
    return points


def visualize():
    # Generate the highly asymmetric T-like point cloud
    points_highly_asymmetric = generate_highly_asymmetric_t_structure(4096)

    # Plotting the generated point cloud
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_highly_asymmetric[:, 0], points_highly_asymmetric[:, 1], points_highly_asymmetric[:, 2])
    ax.set_title("Highly Asymmetric T-like Structure Point Cloud")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()


def generate_random_rotation_matrix(theta=None, phi=None, z=None):
    """
    Generates a random 3D rotation matrix.
    """
    if theta is None:
        theta = np.random.uniform(0, 2 * np.pi)
    if phi is None:
        phi = np.random.uniform(0, 2 * np.pi)
    if z is None:
        z = np.random.uniform(0, 2 * np.pi)

    # Rotation matrices around the x, y, and z axes
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(theta), -np.sin(theta)],
                    [0, np.sin(theta), np.cos(theta)]])

    R_y = np.array([[np.cos(phi), 0, np.sin(phi)],
                    [0, 1, 0],
                    [-np.sin(phi), 0, np.cos(phi)]])

    R_z = np.array([[np.cos(z), -np.sin(z), 0],
                    [np.sin(z), np.cos(z), 0],
                    [0, 0, 1]])

    R = np.dot(R_z, np.dot(R_y, R_x))
    return torch.tensor(R, dtype=torch.float32)


def infer_points(point_cloud, model):
    # Generate a random rotation matrix and apply it to the point cloud
    mat = torch.stack([generate_random_rotation_matrix() for _ in range(point_cloud.shape[0])]).to(device)
    rotated_point_cloud = torch.bmm(mat, point_cloud)

    # Recover the rotation matrix using TNet
    restored_points, predicted_mat = model(rotated_point_cloud)
    return restored_points, predicted_mat, mat


def optimize_pointnet(device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    fig = plt.figure(figsize=(20, 20))
    # Generate the highly asymmetric T-like point cloud
    point_cloud = generate_highly_asymmetric_t_structure(256).transpose(0, 1)  # Shape: [3, P]
    point_cloud = point_cloud.unsqueeze(0).repeat(32, 1, 1)  # Shape: [N, 3, P]
    point_cloud = point_cloud.to(device)
    # Create a TNet model
    model = Tnet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()
    mat_ref = torch.stack([generate_random_rotation_matrix() for _ in range(point_cloud.shape[0])]).to(device)
    valid_rotated_point_cloud = torch.bmm(mat_ref, point_cloud)
    row = 0
    for epoch in range(1001):  # Number of epochs
        optimizer.zero_grad()
        model.train()  # TRAINING MODE to enable dropout!
        # Generate a random rotation matrix and apply it to the point cloud

        mat = torch.stack([generate_random_rotation_matrix() for _ in range(point_cloud.shape[0])]).to(device)
        rotated_point_cloud = torch.bmm(mat, point_cloud)

        # Recover the rotation matrix using TNet
        restored_points, predicted_mat = model(rotated_point_cloud)

        # Calculate the loss (difference between predicted and actual rotation matrices)
        loss = loss_fn(predicted_mat, mat.transpose(-1, -2))  # Adjust the repeat to match batch size

        # Backpropagation
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

        with torch.no_grad():
            if epoch % 100 == 0:
                model.eval()  # EVALUATION MODE to disable dropout!
                restored_points, predicted_mat_val = model(valid_rotated_point_cloud)
                # Move the points back to the CPU
                restored_points = restored_points.cpu().detach().numpy()
                restored_points = restored_points.transpose(0, 2, 1)

                # Plotting the generated point cloud

                for idx in range(10):
                    current_points = restored_points[idx, ...].reshape(-1, 3)
                    ax = fig.add_subplot(11, 10, row*10+idx+1, projection='3d')
                    ax.scatter(current_points[:, 0], current_points[:, 1], current_points[:, 2])
                    # ax.set_xlabel("X")
                    # ax.set_ylabel("Y")
                    # ax.set_zlabel("Z")
                ax.set_title(f"Epoch: {epoch}")
                row += 1
            # plt.savefig(f"__epoch_{epoch:04d}.png")
    # plt.show()
    plt.suptitle("TNET optimization process toy example")
    plt.savefig("TNET_optimization.png")
    plt.close()


if __name__ == "__main__":
    visualize()
    # optimize_pointnet()
