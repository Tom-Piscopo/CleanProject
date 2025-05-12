#! /home/tom/.virtualenvs/fea/bin/python3
''' This is my docustring '''
import numpy as np  # type: ignore
import meshio  # type: ignore
import pandas as pd  # type: ignore
import random
import matplotlib.pyplot as plt  # type: ignore


class XFEMCantileverBeam:
    def __init__(self, length, height, thickness, num_elements_x,
                 num_elements_y, youngs_modulus, shear_modulus, poisson_ratio):
        self.length = length
        self.height = height
        self.thickness = thickness
        self.num_elements_x = num_elements_x
        self.num_elements_y = num_elements_y
        self.youngs_modulus = youngs_modulus
        self.shear_modulus = shear_modulus
        self.poisson_ratio = poisson_ratio
        self.nodes = None
        self.elements = None
        self.global_stiffness_matrix = None
        self.force_vector = None
        self.u_static = None
        # Extended FEM variables
        self.is_cracked = False
        self.crack_start = []
        self.crack_end = []
        self.crack_vector = []
        self.intersected_elements = []
        self.enriched_dofs = []

    def generate_mesh(self):
        x = np.linspace(0, self.length, self.num_elements_x + 1)
        y = np.linspace(0, self.height, self.num_elements_y + 1)
        # Creating nodes
        nodes = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
        # Create elements (connectivity)
        elements = []
        for i in range(self.num_elements_x):
            for j in range(self.num_elements_y):
                n1 = j + i * (self.num_elements_y + 1)
                n2 = n1 + self.num_elements_y + 1
                elements.append([n1, n2, n2 + 1, n1 + 1])

        self.nodes = nodes
        self.elements = np.array(elements, dtype=int)

    def apply_boundary_conditions(self, support_type="cantilever"):
        """
        Apply boundary conditions based on the specified support type.

        Parameters:
        - support_type (str): Type of boundary condition.
          Options:
            - "cantilever": Fixed at x = 0 (both u and v constrained)
            - "simply": Simply supported at x = 0 and x = L (v constrained, u free)
        """
        num_nodes = self.nodes.shape[0]
        self.force_vector = np.zeros(2 * num_nodes)

        if support_type == "cantilever":
            # Find nodes on the left side (x = 0)
            left_nodes = np.where(self.nodes[:, 0] == 0)[0]
            constrained_dofs = []
            for node in left_nodes:
                constrained_dofs.extend([2 * node, 2 * node + 1])  # Fix both u and v

        elif support_type == "simply":
            # Find nodes at x = 0 and x = L
            left_nodes = np.where(self.nodes[:, 0] == 0)[0]
            right_nodes = np.where(self.nodes[:, 0] == self.length)[0]
            constrained_dofs = []

            # Constrain vertical displacement (v) at both ends
            for node in left_nodes:
                constrained_dofs.append(2 * node + 1)  # Fix v at x = 0
            for node in right_nodes:
                constrained_dofs.append(2 * node + 1)  # Fix v at x = L

        else:
            raise ValueError("Invalid support type. Choose 'cantilever' or 'simply'.")

        # Modify global stiffness matrix for fixed DOFs
        for dof in constrained_dofs:
            self.global_stiffness_matrix[dof, :] = 0
            self.global_stiffness_matrix[:, dof] = 0
            self.global_stiffness_matrix[dof, dof] = 1  # Set diagonal to 1 for stability
            self.force_vector[dof] = 0  # Set force at fixed DOFs to zero

    def compute_element_stiffness(self, element_nodes, element_index):
        """
        Compute the stiffness matrix for a single element, including XFEM
        enrichment.

        Parameters:
        - element_nodes: Coordinates of the element nodes (shape: (4, 2)).
        - element_index: Index of the element in the mesh.
        Returns:
        - Element stiffness matrix, including standard and enriched
        contributions.
        """
        # Element properties
        E = self.youngs_modulus
        PR = self.poisson_ratio
        t = self.thickness

        # Plane stress elasticity matrix
        D = (E / (1 - PR ** 2)) * np.array([
            [1, PR, 0],
            [PR, 1, 0],
            [0, 0, (1 - PR) / 2]
        ])

        # Quadrilateral shape functions and integration points
        gauss_points = [-1 / np.sqrt(3), 1 / np.sqrt(3)]
        # 2-point Gauss quadrature
        num_dofs = 8  # 2 DOFs per node for standard FEM
        # Check if the element is enriched
        if self.is_cracked:
            is_intersected = element_index in self.intersected_elements
            is_crack_tip = element_index == self.find_crack_tip_element()
            if is_intersected:
                num_dofs += 4  # Add 4 enriched DOFs for Heaviside function
            if is_crack_tip:
                num_dofs += 16
                # Add 16 enriched DOFs for crack-tip functions
                # (4 nodes × 4 terms each)

        stiffness_matrix = np.zeros((num_dofs, num_dofs))

        for xi in gauss_points:
            for eta in gauss_points:
                # Compute Jacobian and B matrix for standard FEM
                J = self.compute_jacobian(element_nodes, xi, eta)
                detJ = np.linalg.det(J)
                J_inv = np.linalg.inv(J)
                B_standard = self.compute_B_matrix(J_inv, xi, eta)
                # Add standard contribution
                stiffness_matrix[:8, :8] += t * detJ * \
                                            (B_standard.T @ D @ B_standard)
                if self.is_cracked:
                    # Add enrichment contributions
                    start_idx = 8
                    if is_intersected:
                        B_heaviside = self.compute_B_heaviside(J_inv, xi, eta)
                        stiffness_matrix[:8, start_idx:start_idx + 4] += t * detJ \
                                                                         * (B_standard.T @ D @ B_heaviside)
                        stiffness_matrix[start_idx:start_idx + 4, :8] += t * detJ \
                                                                         * (B_heaviside.T @ D @ B_standard)
                        stiffness_matrix[start_idx:start_idx + 4,
                        start_idx:start_idx + 4] += t * detJ \
                                                    * (B_heaviside.T @ D @ B_heaviside)
                        start_idx += 4

                    if is_crack_tip:
                        B_crack_tip = self.compute_B_crack_tip(J_inv, xi, eta)
                        stiffness_matrix[:8, start_idx:start_idx + 16] += t * detJ \
                                                                          * (B_standard.T @ D @ B_crack_tip)
                        stiffness_matrix[start_idx:start_idx + 16, :8] += t * detJ \
                                                                          * (B_crack_tip.T @ D @ B_standard)
                        stiffness_matrix[start_idx:start_idx + 16,
                        start_idx:start_idx + 16] += t * detJ * (
                                B_crack_tip.T @ D @ B_crack_tip)

        return stiffness_matrix

    def compute_B_matrix(self, J_inv, xi, eta):
        """
        Compute the strain-displacement matrix (B).
        """
        # Derivatives of shape functions in natural coordinates
        dN_dxi = np.array([
            [-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)],
            [-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)]
        ]) * 0.25
        # Convert to global coordinates
        dN_dx = J_inv @ dN_dxi
        B = np.zeros((3, 8))
        B[0, 0::2] = dN_dx[0, :]
        B[1, 1::2] = dN_dx[1, :]
        B[2, 0::2] = dN_dx[1, :]
        B[2, 1::2] = dN_dx[0, :]
        return B

    def compute_B_heaviside(self, J_inv, xi, eta):
        """
        Compute the B matrix for the Heaviside enrichment.
        """
        dN_dxi = np.array([
            [-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)],
            [-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)]
        ]) * 0.25

        dN_dx = J_inv @ dN_dxi
        B_heaviside = np.zeros((3, 4))
        B_heaviside[0, :] = dN_dx[0, :]
        B_heaviside[1, :] = dN_dx[1, :]
        B_heaviside[2, :] = dN_dx[1, :]
        # Modify as needed for Heaviside enrichment
        return B_heaviside

    def compute_B_crack_tip(self, J_inv, xi, eta):
        """
        Compute the B matrix for the crack-tip enrichment
        Parameters:
        - J_inv: Inverse Jacobian matrix.
        - xi, eta: Natural coordinates for the Gauss point.

        Returns:
        - B_crack_tip: Strain-displacement matrix for crack-tip enrichment
        (shape: (3, 12)).
        """
        # Derivatives of shape functions in natural coordinates
        dN_dxi = np.array([
            [-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)],
            [-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)]
        ]) * 0.25

        # Convert to global coordinates
        dN_dx = J_inv @ dN_dxi
        # Crack-tip enrichment functions (4 terms per node for asymptotics)
        # Example terms: sqrt(r)*cos(theta/2), sqrt(r)*sin(theta/2), ...
        num_enriched_dofs_per_node = 4
        num_nodes = dN_dx.shape[1]  # 4 nodes in a quadrilateral element
        B_crack_tip = np.zeros((3, num_enriched_dofs_per_node * num_nodes))

        for i in range(num_nodes):
            # Compute enriched terms (placeholder for actual
            # crack-tip functions)
            # Replace with actual r, theta calculations if needed
            r = np.sqrt(xi ** 2 + eta ** 2)  # Radial distance (approximate)
            theta = np.arctan2(eta, xi)  # Angle (approximate)

            # Enrichment derivatives (replace with actual formulas as needed)
            enriched_terms = np.array([
                r ** 0.5 * np.cos(theta / 2),  # Example term 1
                r ** 0.5 * np.sin(theta / 2),  # Example term 2
                r ** 0.5 * np.sin(theta / 2) * np.cos(theta),  # Example term 3
                r ** 0.5 * np.cos(theta / 2) * np.sin(theta)  # Example term 4
            ])

            # Map enriched terms to B matrix
            B_crack_tip[0, i * num_enriched_dofs_per_node:(i + 1) *
                                                          num_enriched_dofs_per_node] = dN_dx[0, i] \
                                                                                        * enriched_terms
            B_crack_tip[1, i * num_enriched_dofs_per_node:(i + 1) *
                                                          num_enriched_dofs_per_node] = dN_dx[1, i] \
                                                                                        * enriched_terms
            B_crack_tip[2, i * num_enriched_dofs_per_node:(i + 1) *
                                                          num_enriched_dofs_per_node] = dN_dx[0, i] \
                                                                                        * enriched_terms + dN_dx[
                                                                                            1, i] * enriched_terms

            return B_crack_tip

    def compute_jacobian(self, element_nodes, xi, eta):
        """
        Compute the Jacobian matrix for an element.
        """
        # Shape function derivatives in natural coordinates
        dN_dxi = np.array([
            [-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)],
            [-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)]
        ]) * 0.25

        # Compute Jacobian
        J = dN_dxi @ element_nodes
        return J

    def assemble_global_stiffness(self):
        """
        Assemble the global stiffness matrix, including XFEM enrichment for
        intersected elements.
        """
        # Calculate total DOFs
        num_nodes = len(self.nodes)
        standard_dofs = 2 * num_nodes  # 2 DOFs (u_x, u_y) per node
        enriched_dofs = len(self.enriched_dofs)
        # 1 Heaviside DOF or multiple crack-tip DOFs per enriched node
        total_dofs = standard_dofs + enriched_dofs

        # Initialize global stiffness matrix and force vector
        self.global_stiffness_matrix = np.zeros((total_dofs, total_dofs))
        self.force_vector = np.zeros(total_dofs)

        # Map enriched DOFs to global indices
        dof_map = {node: idx + standard_dofs for idx,
        node in enumerate(self.enriched_dofs)}

        # Assemble stiffness matrix for each element
        for elem_idx, element in enumerate(self.elements):
            element_nodes = self.nodes[element]
            k_element = self.compute_element_stiffness(element_nodes, elem_idx)

            # Global DOF mapping
            global_dofs = []
            # Add standard DOFs for all nodes in the element
            for node in element:
                global_dofs.extend([2 * node, 2 * node + 1])
            if self.is_cracked:
                # Handle enrichment for intersected elements
                if elem_idx in self.intersected_elements:
                    for node in element:
                        if node in self.enriched_dofs:
                            enriched_idx = dof_map[node]
                            global_dofs.append(enriched_idx)
                            # Add 1 Heaviside DOF per node

                # Handle crack-tip enrichment
                if elem_idx == self.find_crack_tip_element():
                    for node in element:
                        if node in self.enriched_dofs:
                            enriched_idx = dof_map[node]
                            for i in range(4):
                                # Add 4 DOFs per node for crack-tip enrichment
                                global_dofs.append(enriched_idx + i)

                # Ensure global_dofs and k_element sizes match
                if len(global_dofs) != k_element.shape[0]:
                    raise ValueError(f"Mismatch between global_dofs \
                            ({len(global_dofs)}) and k_element size \
                            ({k_element.shape[0]}) for element {elem_idx}")

            # Add element contributions to global stiffness matrix
            for i, global_i in enumerate(global_dofs):
                for j, global_j in enumerate(global_dofs):
                    self.global_stiffness_matrix[global_i, global_j] += \
                        k_element[i, j]

    def solve_static_system(self):
        """
        Solve the static system, including enriched DOFs for XFEM.

        Returns:
        - u_static: Displacement vector (standard + enriched DOFs).
        """
        # Check if the global stiffness matrix and force vector are assembled
        if self.global_stiffness_matrix is None or self.force_vector is None:
            raise ValueError("Global stiffness matrix or force vector is not\
                    assembled.")

        # Extend the force vector to match the augmented global matrix size
        num_total_dofs = self.global_stiffness_matrix.shape[0]
        if len(self.force_vector) < num_total_dofs:
            extended_force_vector = np.zeros(num_total_dofs)
            extended_force_vector[:len(self.force_vector)] = self.force_vector
            self.force_vector = extended_force_vector

        # Solve the augmented system of equations
        self.u_static = np.linalg.solve(self.global_stiffness_matrix,
                                        self.force_vector)

        return self.u_static

    def apply_concentrated_load(self, load, load_location):
        """
        Apply a concentrated load to the beam.
        """
        # Find the node closest to the load location
        distances = np.linalg.norm(self.nodes - load_location, axis=1)
        load_node = np.argmin(distances)

        # Apply the load to the corresponding DOFs
        self.force_vector[2 * load_node] += load[0]  # x-direction
        self.force_vector[2 * load_node + 1] += load[1]  # y-direction

    # EXTENDED FINITE ELEMENT METHODS

    def add_surface_crack(self, start, end):
        """
        Add a surface crack to the beam.

        Parameters:
        - start: (x, y) tuple, starting point of the crack.
        - end: (x, y) tuple, ending point of the crack.
        """
        # Store the crack geometry
        self.is_cracked = True
        self.crack_start = np.array(start)
        self.crack_end = np.array(end)
        self.crack_vector = self.crack_end - self.crack_start

        # Identify intersected elements
        intersected_elements = []
        for idx, element in enumerate(self.elements):
            element_nodes = self.nodes[element]
            if self.is_element_intersected_by_crack(element_nodes):
                intersected_elements.append(idx)

        # Store the intersected elements
        self.intersected_elements = intersected_elements

    def is_element_intersected_by_crack(self, element_nodes):
        """
        Check if a crack intersects an element.

        Parameters:
        - element_nodes: Numpy array of shape (4, 2), coordinates of the
        element nodes.

        Returns:
        - Boolean indicating whether the element is intersected by the crack.
        """
        # Check for intersection using line-segment intersection tests
        crack_start, crack_end = self.crack_start, self.crack_end
        for i in range(4):
            node1 = element_nodes[i]
            node2 = element_nodes[(i + 1) % 4]
            if self.lines_intersect(crack_start, crack_end, node1, node2):
                return True
        return False

    def lines_intersect(self, p1, p2, q1, q2):
        """
        Check if two line segments intersect.

        Parameters:
        - p1, p2: Endpoints of the first line segment.
        - q1, q2: Endpoints of the second line segment.

        Returns:
        - Boolean indicating intersection.
        """

        def ccw(a, b, c):
            return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) \
                * (c[0] - a[0])

        return (ccw(p1, q1, q2) != ccw(p2, q1, q2)) and (ccw(p1, p2, q1) !=
                                                         ccw(p1, p2, q2))

    def add_enrichment(self):
        """
        Add XFEM enrichment for the crack, ensuring consistent enrichment for
        all intersected elements.
        """
        if self.is_cracked:
            self.enriched_dofs = []  # Reset enriched DOFs
            crack_tip_element = self.find_crack_tip_element()

            # Add crack-tip enrichment
            if crack_tip_element is not None:
                for node in self.elements[crack_tip_element]:
                    if node not in self.enriched_dofs:
                        self.enriched_dofs.append(node)

            # Add Heaviside enrichment for all other intersected elements
            for elem_idx in self.intersected_elements:
                if elem_idx == crack_tip_element:
                    continue  # Skip crack-tip element
                for node in self.elements[elem_idx]:
                    if node not in self.enriched_dofs:
                        self.enriched_dofs.append(node)
        else:
            pass

    def find_crack_tip_element(self):
        """
        Identify the element containing the crack tip.
        The crack tip is defined as the end of the crack geometry.

        Returns:
            crack_tip_element: The index of the crack-tip element.
        """
        crack_tip = self.crack_end  # The end point of the crack
        for elem_idx, element in enumerate(self.elements):
            element_nodes = self.nodes[element]
            if self.is_point_inside_element(crack_tip, element_nodes):
                return elem_idx
        return None

    def is_point_inside_element(self, point, element_nodes):
        """
        Check if a point lies inside an element using barycentric coordinates.
        """
        x, y = point
        x_coords = element_nodes[:, 0]
        y_coords = element_nodes[:, 1]
        area = 0.5 * abs(
            x_coords[0] * (y_coords[1] - y_coords[3]) +
            x_coords[1] * (y_coords[2] - y_coords[0]) +
            x_coords[2] * (y_coords[3] - y_coords[1]) +
            x_coords[3] * (y_coords[0] - y_coords[2])
        )
        for i in range(4):
            node_area = 0.5 * abs(
                x_coords[i] * (y_coords[(i + 1) % 4] - y) +
                x_coords[(i + 1) % 4] * (y - y_coords[i]) +
                x * (y_coords[i] - y_coords[(i + 1) % 4])
            )
            if node_area > area:
                return False
        return True

    def compute_node_stresses(self):
        """
        Compute stresses at each node of the beam based on the current displacement field.

        Returns:
            stresses: A numpy array of shape (num_nodes, 3), where each row contains
                    [sigma_xx, sigma_yy, sigma_xy] for a node.
        """
        if self.u_static is None:
            raise ValueError("Displacement field (u_static) not computed. Solve the system first.")

        num_nodes = len(self.nodes)
        stresses = np.zeros((num_nodes, 3))  # To store [sigma_xx, sigma_yy, sigma_xy] for each node

        # Loop through each element to calculate strains and stresses
        for elem_idx, element in enumerate(self.elements):
            element_nodes = self.nodes[element]  # Coordinates of the element nodes

            # Extract displacements for the element nodes
            element_dofs = []
            for node in element:
                element_dofs.extend([2 * node, 2 * node + 1])
            element_displacements = self.u_static[element_dofs]

            # Quadrature points (2-point Gauss quadrature)
            gauss_points = [-1 / np.sqrt(3), 1 / np.sqrt(3)]

            # Loop through Gauss points to compute strain and stress
            for xi in gauss_points:
                for eta in gauss_points:
                    # Compute Jacobian and B matrix
                    J = self.compute_jacobian(element_nodes, xi, eta)
                    J_inv = np.linalg.inv(J)
                    B = self.compute_B_matrix(J_inv, xi, eta)

                    # Compute strain
                    strain = B @ element_displacements

                    # Compute stress using Hooke's law (plane stress condition)
                    D = (self.youngs_modulus / (1 - self.poisson_ratio ** 2)) * np.array([
                        [1, self.poisson_ratio, 0],
                        [self.poisson_ratio, 1, 0],
                        [0, 0, (1 - self.poisson_ratio) / 2]
                    ])
                    stress = D @ strain

                    # Add stress contribution to the element nodes (average later)
                    for i, node in enumerate(element):
                        stresses[node] += stress

        # Average the stresses at each node
        node_counts = np.zeros(num_nodes)
        for element in self.elements:
            for node in element:
                node_counts[node] += 1

        for i in range(num_nodes):
            if node_counts[i] > 0:
                stresses[i] /= node_counts[i]

        return stresses

    '''def compute_node_stresses(self):
        """
        Compute stresses at each node of the beam based on the current displacement field.

        Returns:
            stresses: A numpy array of shape (num_nodes, 3), where each row contains
                    [sigma_xx, sigma_yy, sigma_xy] for a node.
        """
        if self.u_static is None:
            raise ValueError("Displacement field (u_static) not computed. Solve the system first.")

        num_nodes = len(self.nodes)
        stresses = np.zeros((num_nodes, 3))  # To store [sigma_xx, sigma_yy, sigma_xy] for each node
        node_counts = np.zeros(num_nodes)

        # Loop through each element to calculate strains and stresses
        for elem_idx, element in enumerate(self.elements):
            element_nodes = self.nodes[element]  # Coordinates of the element nodes
            element_dofs = [2 * node for node in element] + [2 * node + 1 for node in element]
            element_displacements = self.u_static[element_dofs]

            # Quadrature points (2-point Gauss quadrature)
            gauss_points = [-1 / np.sqrt(3), 1 / np.sqrt(3)]

            for xi in gauss_points:
                for eta in gauss_points:
                    # Compute Jacobian and B matrix
                    J = self.compute_jacobian(element_nodes, xi, eta)
                    J_inv = np.linalg.inv(J)
                    B = self.compute_B_matrix(J_inv, xi, eta)

                    # Compute strain
                    strain = B @ element_displacements

                    # Compute stress using Hooke's law (plane stress condition)
                    D = (self.youngs_modulus / (1 - self.poisson_ratio**2)) * np.array([
                        [1, self.poisson_ratio, 0],
                        [self.poisson_ratio, 1, 0],
                        [0, 0, (1 - self.poisson_ratio) / 2]
                    ])
                    stress = D @ strain

                    # Add stress contribution to element nodes
                    for i, node in enumerate(element):
                        stresses[node] += stress
                        node_counts[node] += 1  # Keep track of contributions

        # Average stresses at each node
        for i in range(num_nodes):
            if node_counts[i] > 0:
                stresses[i] /= node_counts[i]

        return stresses'''

    def compute_crack_tip_sif(self):
        """
        Compute Stress Intensity Factors (SIFs) at the crack tip.
        Mode I (K_I) and Mode II (K_II) are calculated based on crack-tip enrichment DOFs.

        Returns:
            sif_results: A dictionary mapping the crack-tip node to its respective K_I and K_II values.
        """

        # Material properties
        E = self.youngs_modulus
        nu = self.poisson_ratio
        mu = E / (2 * (1 + nu))  # Shear modulus

        # Distance from crack tip (r) - approximate it as half the element size
        element_size_x = self.length / self.num_elements_x
        element_size_y = self.height / self.num_elements_y
        r = np.sqrt((element_size_x / 2) ** 2 + (element_size_y / 2) ** 2)

        # Use self.crack_end for crack tip coordinates
        crack_tip_coords = self.crack_end

        # Find the crack-tip element
        crack_tip_element = self.find_crack_tip_element()
        if crack_tip_element is None:
            raise ValueError("Crack-tip element not found. Ensure crack geometry is defined correctly.")

        # Find the crack-tip node (nearest node to crack tip within the element)
        element_nodes = self.elements[crack_tip_element]
        distances = np.linalg.norm(self.nodes[element_nodes] - crack_tip_coords, axis=1)
        crack_tip_node = element_nodes[np.argmin(distances)]  # Node closest to crack tip

        # Verify that crack-tip node is in enriched DOFs list
        if crack_tip_node not in self.enriched_dofs:
            raise ValueError(f"Crack-tip node {crack_tip_node} is not in enriched DOFs list.")

        # Start of enriched DOFs in u_static
        enriched_dof_start = 2 * len(self.nodes)

        # Extract enriched DOFs
        enriched_idx = enriched_dof_start + self.enriched_dofs.index(crack_tip_node) * 4
        enriched_dofs = self.u_static[enriched_idx:enriched_idx + 4]

        # Validate enriched DOFs
        if len(enriched_dofs) != 4:
            raise ValueError(f"Enriched DOFs for crack-tip node {crack_tip_node} are invalid: {enriched_dofs}")

        # Extract standard DOFs (u_x, u_y)
        standard_dofs = self.u_static[2 * crack_tip_node: 2 * crack_tip_node + 2]

        # Combine standard and enriched DOFs
        dofs = np.concatenate((standard_dofs, enriched_dofs))

        # Validate combined DOFs
        if len(dofs) != 6:
            raise ValueError(f"Combined DOFs for crack-tip node {crack_tip_node} are invalid: {dofs}")

        # Debugging DOFs for SIF calculation

        # Compute SIFs
        K_I = 2 * mu * np.sqrt(np.pi * r) * dofs[0]  # Expected enrichment term for K_I
        K_II = 2 * mu * np.sqrt(np.pi * r) * dofs[1]  # Expected enrichment term for K_II

        # Compute SIFs
        K_I = 2 * mu * np.sqrt(np.pi * r) * dofs[0]  # r^(1/2) cos(theta/2)
        K_II = 2 * mu * np.sqrt(np.pi * r) * dofs[1]  # r^(1/2) sin(theta/2)

        # Return results
        sif_results = {"K_I": K_I, "K_II": K_II}
        return sif_results

    import numpy as np


def compute_gauss_point_stresses(self):
    """
    Compute the stresses at Gauss points for each element in the beam.

    Returns:
        gauss_stresses: List of arrays, where each array corresponds to the stresses
                        at the Gauss points of an element. Each array is of shape (num_gauss_points, 3),
                        where the 3 components are [sigma_xx, sigma_yy, tau_xy].
    """
    gauss_stresses = []

    # Material properties
    E = self.youngs_modulus
    PR = self.poisson_ratio
    D = (E / (1 - PR ** 2)) * np.array([
        [1, PR, 0],
        [PR, 1, 0],
        [0, 0, (1 - PR) / 2]
    ])

    # Gauss points and weights for 2-point Gauss quadrature
    gauss_points = [-1 / np.sqrt(3), 1 / np.sqrt(3)]
    weights = [1, 1]

    # Loop through elements
    for elem_idx, element in enumerate(self.elements):
        element_nodes = self.nodes[element]
        element_displacements = self.u_static[np.array([2 * n for n in element] + [2 * n + 1 for n in element])]

        # List to store stresses for this element's Gauss points
        element_gauss_stresses = []

        # Loop through Gauss points
        for xi in gauss_points:
            for eta in gauss_points:
                # Compute the Jacobian, its determinant, and the inverse
                J = self.compute_jacobian(element_nodes, xi, eta)
                detJ = np.linalg.det(J)
                J_inv = np.linalg.inv(J)

                # Compute the strain-displacement matrix B
                B = self.compute_B_matrix(J_inv, xi, eta)

                # Compute the strain vector
                strain = B @ element_displacements

                # Compute the stress vector using the material stiffness matrix D
                stress = D @ strain

                # Append the stress at this Gauss point
                element_gauss_stresses.append(stress)

        # Convert to array and append to the global list
        gauss_stresses.append(np.array(element_gauss_stresses))

    return gauss_stresses

    # MISCELLANEOUS METHODS


def plot_stresses_along_height(beam, stresses, crack_tip_height, stress_component=0):
    """
    Plot stresses along the beam at the same height as the crack tip.

    Parameters:
        beam: XFEMCantileverBeam object containing the mesh and nodes.
        stresses: Numpy array of shape (num_nodes, 3), stresses at each node.
        crack_tip_height: Height (y-coordinate) of the crack tip.
        stress_component: Index of the stress component to plot:
                          0 -> sigma_xx, 1 -> sigma_yy, 2 -> sigma_xy.
    """
    # Select nodes at the crack tip height
    nodes_at_height = [
        (i, node) for i, node in enumerate(beam.nodes) if np.isclose(node[1], crack_tip_height, atol=1e-6)
    ]

    if not nodes_at_height:
        print(f"No nodes found at height {crack_tip_height}. Check your mesh or crack height.")
        return

    # Sort nodes by x-coordinate
    nodes_at_height.sort(key=lambda x: x[1][0])  # Sort by x-coordinate
    node_indices = [item[0] for item in nodes_at_height]
    x_coords = [item[1][0] for item in nodes_at_height]
    stress_values = [stresses[i, stress_component] for i in node_indices]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_coords, stress_values, marker='o', linestyle='-', label=f"Stress Component {stress_component}")
    plt.title(f"Stress Distribution Along Height y = {crack_tip_height}")
    plt.xlabel("X-Coordinate")
    plt.ylabel(f"Stress Component {stress_component} (Pa)")
    plt.grid(True)
    plt.legend()
    plt.show()

    def get_properties(self):
        """
        Return beam properties in a dict
        """
        properties = {
            "Length": self.length,
            "Height": self.height,
            "Thickness": self.thickness,
            "Young's Modulus": self.youngs_modulus,
            "Shear Modulus": self.shear_modulus,
            "Poisson Ratio": self.poisson_ratio
        }
        return properties

    def export_mesh(self, filename="beam_mesh.vtk"):
        """
        Export the mesh to a VTK file using meshio
        """
        if self.nodes is None or self.elements is None:
            raise ValueError("Mesh not generated. Call 'generate_mesh' first.")

        # Convert to meshio format
        cells = [("quad", self.elements)]
        mesh = meshio.Mesh(points=self.nodes, cells=cells)

        # Write to a file
        meshio.write(filename, mesh)
        print(f"Mesh saved to {filename}")

    def write_deformed_mesh(self, displacements, filename="deformed_beam.vtk"):
        """
        Write the deformed mesh to a VTK file.

        Parameters:
        - displacements: Numpy array of shape (2 * num_nodes,)
                        Contains displacement values
                        [u1, v1, u2, v2, ..., uN, vN].
        - filename: String, name of the output VTK file.
        """
        if self.nodes is None or self.elements is None:
            raise ValueError("Mesh not generated. Call `generate_mesh` first.")

        num_nodes = self.nodes.shape[0]

        # Reshape displacements to (num_nodes, 2) for [u, v]
        node_displacements = displacements.reshape((num_nodes, 2))

        # Compute deformed node coordinates
        deformed_nodes = self.nodes + node_displacements

        # Convert to meshio format
        cells = [("quad", self.elements)]
        point_data = {"displacement": node_displacements}
        mesh = meshio.Mesh(points=deformed_nodes, cells=cells,
                           point_data=point_data)

        # Write to file
        meshio.write(filename, mesh)
        print(f"Deformed mesh saved to {filename}")


def analytical_stress_vertical(height, thickness, P, crack_tip_y, num_points):
    """
    Compute the analytical stress distribution for a cantilever beam with a surface crack,
    traversing vertically from just beyond the crack tip to the bottom of the beam.

    Parameters:
    - height: Beam height (m).
    - thickness: Beam thickness (m).
    - P: Applied tensile load (N).
    - crack_tip_x: X-coordinate of the crack tip (m).
    - crack_tip_y: Y-coordinate of the crack tip (m).
    - num_points: Number of points to evaluate along the vertical height.

    Returns:
    - y_coords: Y-coordinates of the stress points (moving downward from the crack tip).
    - stresses: Analytical stress values at those y-coordinates.
    """
    b = height  # Beam height
    h = thickness
    A = b * h  # Cross-sectional area

    # Generate y-coordinates: from just below the crack tip to the bottom of the beam
    y_coords = np.linspace(1 - 1e-4, 0, num_points)  # Avoid exact crack tip singularity

    # Initialize stress array
    stresses = np.zeros_like(y_coords)

    # Stress intensity factor
    a = crack_tip_y  # Crack depth
    K_I = (P * np.sqrt(a) / (b * h * np.sqrt(np.pi))) * (
                1.99 - 0.41 * (a / b) + 18.70 * (a / b) ** 2 - 38.48 * (a / b) ** 3 + 53.85 * (a / b) ** 4)
    # K_I = (P / A) * np.sqrt(np.pi * a)

    # Threshold for near-field
    r_threshold = 5 * a  # 5 times the crack depth

    for i, y in enumerate(y_coords):
        if y < crack_tip_y:
            stresses[i] = 0
        else:
            r = abs(y - crack_tip_y)  # Distance from crack tip

            if r < r_threshold:  # Near-field region
                stresses[i] = K_I / np.sqrt(2 * np.pi * r)
            else:  # Far-field region
                stresses[i] = P / (A)

    return y_coords, stresses


def analytical_stress_distribution(length, height, thickness, P, crack_tip_x, crack_depth, num_points):
    """
    Compute the analytical stress distribution for a cantilever beam with a surface crack.

    Parameters:
    - length: Beam length (m).
    - height: Beam height (m).
    - thickness: Beam thickness (m).
    - P: Applied tensile load (N).
    - crack_tip_x: X-coordinate of the crack tip (m).
    - crack_depth: Crack depth (m).
    - num_points: Number of points to evaluate along the beam length.

    Returns:
    - x_coords: X-coordinates of the stress points.
    - stresses: Analytical stress values.
    """
    b = height  # Height of the beam
    h = thickness
    a = crack_depth
    c = b - a

    # Generate x-coordinates for the distribution
    x_coords = np.linspace(0, length, num_points)

    # Initialize stress array
    stresses = np.zeros_like(x_coords)

    # Stress intensity factor for the crack
    # K_I = (P / (b * h)) * np.sqrt(np.pi * a)
    K_I = (P * np.sqrt(a) / (b * h * np.sqrt(np.pi))) * (
                1.99 - 0.41 * (a / b) + 18.70 * (a / b) ** 2 - 38.48 * (a / b) ** 3 + 53.85 * (a / b) ** 4)

    for i, x in enumerate(x_coords):
        r = abs(x - crack_tip_x)  # Distance to crack tip

        # Check if we're near the crack tip
        if r < 5 * a and K_I / np.sqrt(2 * np.pi * r) > P / (b * h):  # Threshold for "near-field"
            # Near the crack tip: use stress singularity field
            stresses[i] = K_I / np.sqrt(2 * np.pi * r)
            # print(f"Near-field stress at x = {x}: {stresses[i]}")
        else:
            # Far-field stress for nominal areas (away from crack tip)
            stresses[i] = P / (b * h)
            # print(f"Far-field stress at x = {x}: {stresses[i]}")

    return x_coords, stresses


def plot_stress_comparison(beam, stresses, analytical_x_coords, analytical_stresses, crack_tip_x, crack_tip_height,
                           stress_component=0):
    """
    Compare and plot FE stresses with the analytical stress distribution.

    Parameters:
    - beam: The XFEMCantileverBeam instance.
    - stresses: Numpy array of stresses at each node.
    - analytical_x_coords: X-coordinates for the analytical stress distribution.
    - analytical_stresses: Stress values for the analytical solution.
    - crack_tip_x: X-coordinate of the crack tip.
    - crack_tip_height: Y-coordinate of the crack tip.
    - stress_component: The stress component to plot (0 for σ_xx, 1 for σ_yy, etc.).
    """

    # Get the x-coordinates and y-coordinates of all nodes
    x_coords_all = beam.nodes[:, 0]
    y_coords_all = beam.nodes[:, 1]

    # Select the nodes at the crack tip height (y = crack_tip_height)
    tolerance = 1e-5  # Small tolerance for floating-point comparisons
    row_indices = np.where(np.abs(y_coords_all - crack_tip_height) < tolerance)[0]

    # Filter the x-coordinates and stresses for these nodes
    x_coords = x_coords_all[row_indices]
    fe_stresses = stresses[row_indices, stress_component]  # Extract the desired stress component

    # Ensure x_coords and fe_stresses match
    assert len(x_coords) == len(fe_stresses), "Mismatched dimensions!"

    # Sort the FE stresses and coordinates for smoother plotting
    sorted_indices = np.argsort(x_coords)
    x_coords = x_coords[sorted_indices]
    fe_stresses = fe_stresses[sorted_indices]

    # Plot FE and Analytical Stresses
    plt.figure(figsize=(10, 6))
    plt.plot(x_coords, fe_stresses, label="FE Stress (σ_xx)", marker='o')
    plt.plot(analytical_x_coords, analytical_stresses, label="Analytical Stress", linestyle='--')
    plt.axvline(crack_tip_x, color='red', linestyle='--', label="Crack Tip")
    plt.title("Stress Comparison: FE vs Analytical (σ_xx)")
    plt.xlabel("X-Coordinate (m)")
    plt.ylabel("Stress (Pa)")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_stress_vertical(beam, stresses, analytical_y_coords, analytical_stresses, crack_tip_x, stress_component=1):
    """
    Compare and plot FE stresses with the analytical stress distribution along a vertical profile at the crack tip.

    Parameters:
    - beam: The XFEMCantileverBeam instance.
    - stresses: Numpy array of stresses at each node.
    - analytical_y_coords: Y-coordinates for the analytical stress distribution.
    - analytical_stresses: Stress values for the analytical solution.
    - crack_tip_x: X-coordinate of the crack tip.
    - stress_component: The stress component to plot (0 for σ_xx, 1 for σ_yy, etc.).
    """

    # Get the x-coordinates and y-coordinates of all nodes
    x_coords_all = beam.nodes[:, 0]
    y_coords_all = beam.nodes[:, 1]

    # Select the nodes at the crack tip x-location
    tolerance = 1e-5  # Small tolerance for floating-point comparisons
    column_indices = np.where(np.abs(x_coords_all - crack_tip_x) < tolerance)[0]

    # Get FE stress values at the crack tip x-location
    fe_y_coords = y_coords_all[column_indices]
    fe_stresses = stresses[column_indices, stress_component]  # Extract the desired stress component (σ_yy)

    # Sort for plotting
    sorted_indices = np.argsort(fe_y_coords)
    fe_y_coords = fe_y_coords[sorted_indices]
    fe_stresses = fe_stresses[sorted_indices]

    # Plot FE and Analytical Stresses along vertical profile
    plt.figure(figsize=(6, 8))
    plt.plot(fe_y_coords[::-1], fe_stresses, label="FE Stress (σ_xx)", marker='o')
    plt.plot(analytical_y_coords, analytical_stresses, label="Analytical Stress", linestyle='--')
    plt.axhline(beam.height, color='red', linestyle='--', label="Crack Tip")
    plt.title("Vertical Stress Comparison: FE vs Analytical (σ_xx)")
    plt.xlabel("Y-Coordinate (m)")
    plt.ylabel("Stress (Pa)")
    plt.legend()
    plt.grid(True)
    plt.gca().invert_yaxis()  # Invert y-axis so top of beam is at top of plot
    plt.show()


if __name__ == "__main__":
    # Beam Parameters
    length = 10
    height = 1
    thickness = 0.1
    num_elements_x = 60
    num_elements_y = 20
    E = 10e+7
    G = 3.8e+6
    PR = 0.33

    data = []

    # loading condition
    # Number of iterations
    iter = 5
    for i in range(iter):
        crack_start_x = random.uniform(1.5, 8.5)
        crack_start_y = 0
        crack_end_y = random.uniform(0.11, 0.50)
        P = random.randint(1000, 10000)  # Applied tensile load (N)
        load = np.array([0, -P])
        load_location_x = random.uniform(1.5, 8.5)
        load_location_y = height
        load_location = np.array([load_location_x, load_location_y])

        # Initialize beam
        beam = XFEMCantileverBeam(length, height, thickness, num_elements_x,
                                  num_elements_y, E, G, PR)
        beam.generate_mesh()

        # Define crack geometry
        crack_start = (crack_start_x, crack_start_y)
        crack_end = (crack_start_x, crack_end_y)
        beam.add_surface_crack(crack_start, crack_end)
        beam.add_enrichment()

        # Assembly & Boundary Conditions
        beam.assemble_global_stiffness()
        beam.apply_boundary_conditions("simply")

        # Apply Load & Solve
        beam.apply_concentrated_load(load, load_location)
        u = beam.solve_static_system()

        # Extract displacements
        num_nodes = len(beam.nodes)
        standard_u = u[:2 * num_nodes]  # Extract first 2*num_nodes entries
        standard_u = standard_u.reshape((num_nodes, 2))

        # Identify bottom surface nodes
        bottom_nodes = np.where(beam.nodes[:, 1] == 0)[0]  # Nodes where y = 0
        bottom_displacements = standard_u[bottom_nodes, 1]  # Extract vertical displacements (v)

        # Store results (now storing the full bottom displacement array)
        data.append([crack_start, crack_end, bottom_displacements.tolist()])

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=['crack_start', 'crack_tip', 'bottom_surface_deflections'])

    # Write to CSV
    df.to_csv("test.csv", index=False)

    # Print last 50 bottom surface deflection arrays for verification
    # print("Last 50 bottom surface deflections:")
    # print(df['bottom_surface_deflections'].tail(50))
    """
    # Example usage after solving the system and computing stresses
    crack_tip_height = 0.5  # Replace with the actual crack tip height
    stresses = beam.compute_node_stresses()/4 
    plot_stresses_along_height(beam, stresses, crack_tip_height, stress_component=1)
    # Example Usage
    crack_tip_x = 5.0  # Crack tip location (m)
    crack_depth = 0.5  # Crack depth (m)

    analytical_x_coords, analytical_stresses = analytical_stress_distribution(length, height, thickness, P, crack_tip_x, crack_end_y, num_points=num_elements_x*50)
    analytical_y_coords, analytical_stresses_y = analytical_stress_vertical(height, thickness, P, crack_end_y, num_points=num_elements_y*50)

    # Get the x-coordinates of all nodes
    #x_coords_all = beam.nodes[:, 0]
    #y_coords_all = beam.nodes[:, 1]

    # Select the nodes at the crack tip height (y = 0.5)
    #crack_tip_y = 0.5
    #tolerance = 1e-5  # Small tolerance for floating-point comparisons
    #row_indices = np.where(np.abs(y_coords_all - crack_tip_y) < tolerance)[0]

    # Filter the x-coordinates and stresses for these nodes
    #x_coords = x_coords_all[row_indices]
    #fe_stresses_xx = stresses[row_indices, 0]  # Extract σ_xx for the selected nodes
    #print(beam.compute_crack_tip_sif())

    # Ensure x_coords and fe_stresses_xx match
    #assert len(x_coords) == len(fe_stresses_xx), "Mismatched dimensions!"

    # Plot comparison of FE stresses and analytical stresses
    '''plot_stress_comparison(
        beam=beam, 
        stresses=stresses, 
        analytical_x_coords=analytical_x_coords, 
        analytical_stresses=analytical_stresses,
        crack_tip_x=crack_tip_x,  
        crack_tip_height=0.5, # crack_tip_y,  # Crack tip height
        stress_component=0  # σ_xx
    )'''

    plot_stress_vertical(beam, stresses, analytical_y_coords, analytical_stresses_y, crack_tip_x, stress_component=0)
    """
