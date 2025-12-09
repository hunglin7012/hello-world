# hello-world
 "This repository is for practicing the GitHub Flow."
bcsjbczc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

# ============================================================
# USER SETTINGS (YOU CAN EDIT THESE)
# ============================================================

L      = 20.0        # domain length [m]
T_end  = 300.0       # total time [s]
dx     = 0.2         # spatial step [m]
dt     = 10.0        # time step [s]
U_base = 0.1         # base velocity [m/s]

IC_FILENAME = "initial_conditions.csv"  # make sure this is in the same folder

# store profiles every N time steps (for plots & animation)
STORE_EVERY = 6      # 6 * 10 s = 60 s between stored profiles

# Where to save figures / animations
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# CORE NUMERICAL FUNCTIONS (BACKWARD DIFFERENCES)
# ============================================================

def build_grid(L, T_end, dx, dt):
    """
    Build 1D spatial and temporal grids.
    """
    Nx = int(round(L / dx)) + 1
    Nt = int(round(T_end / dt)) + 1
    x  = np.linspace(0.0, L, Nx)
    t  = np.linspace(0.0, T_end, Nt)
    return x, t, Nx, Nt


def assemble_AB(u_desc, x, dx, dt, t_val=None):
    """
    Assemble diagonal arrays A and B for the bidiagonal system using
    backward differences:

        a_i = 1/dt + u_i / dx
        b_i = -u_i / dx

    u_desc can be:
      - scalar U
      - 1D array u(x_i)
      - callable u(x, t) returning 1D array
    """
    # get u(x) at this time
    if callable(u_desc):
        u_vec = np.asarray(u_desc(x, t_val), dtype=float)
    elif np.isscalar(u_desc):
        u_vec = np.full_like(x, float(u_desc), dtype=float)
    else:
        u_vec = np.asarray(u_desc, dtype=float)
        if u_vec.shape != x.shape:
            raise ValueError("Velocity array must have same shape as x")

    a_i = 1.0 / dt + u_vec / dx
    b_i = -u_vec / dx           # minus sign here

    # A, B correspond to interior nodes i = 1..Nx-1
    A = a_i[1:].copy()
    B = b_i[1:].copy()
    return A, B


def forward_step(theta_old, A, B, dt, t_now, bc_func):
    """
    Perform one time step using sparse forward substitution:

        F[I]     = (1/dt) * theta_old[I+1]
        theta[0] = boundary at x=0
        theta[I] = (F[I-1] - B[I-1] * theta[I-1]) / A[I-1], I = 1..Nx-1
    """
    Nx = theta_old.size
    theta_new = np.zeros_like(theta_old)
    F = np.zeros(Nx - 1, dtype=float)

    # boundary at x = 0
    theta_new[0] = bc_func(t_now)

    # right-hand side
    F[:] = (1.0 / dt) * theta_old[1:]

    # forward substitution along x
    for I in range(1, Nx):
        theta_new[I] = (F[I - 1] - B[I - 1] * theta_new[I - 1]) / A[I - 1]

    return theta_new


def run_model(x, t, theta0, u_desc, bc_func, store_every=1):
    """
    Generic advection solver using the backward-difference scheme.

    u_desc: scalar, array, or callable u(x, t).
    bc_func: function giving theta(t, x=0).
    store_every: store every N-th timestep for plotting/animation.

    Returns:
        times_stored  : (n_store,)
        thetas_stored : (n_store, Nx)
    """
    Nx = x.size
    Nt = t.size
    dx = x[1] - x[0]
    dt = t[1] - t[0]

    theta_old = theta0.copy()

    times_stored  = [t[0]]
    thetas_stored = [theta_old.copy()]

    time_dependent_u = callable(u_desc)

    if not time_dependent_u:
        A, B = assemble_AB(u_desc, x, dx, dt)

    for j in range(1, Nt):
        t_now = t[j]

        if time_dependent_u:
            A, B = assemble_AB(u_desc, x, dx, dt, t_val=t_now)

        theta_new = forward_step(theta_old, A, B, dt, t_now, bc_func)

        if j % store_every == 0:
            times_stored.append(t_now)
            thetas_stored.append(theta_new.copy())

        theta_old[:] = theta_new

    return np.array(times_stored), np.array(thetas_stored)


# ============================================================
# INITIAL CONDITIONS FROM CSV (TEST 2)
# ============================================================

def read_and_interpolate_ic(filename, x):
    """
    Read 'initial_conditions.csv' and interpolate onto model grid x.
    Assumes first column = distance [m], second column = concentration.
    """
    df = pd.read_csv(filename, encoding="latin1")
    xp = df.iloc[:, 0].to_numpy(dtype=float)
    fp = df.iloc[:, 1].to_numpy(dtype=float)
    theta0 = np.interp(x, xp, fp)
    return theta0


# ============================================================
# PLOTTING & ANIMATION HELPERS
# ============================================================

def plot_profiles(x, times, thetas, title_prefix="", save_path=None, show=False):
    plt.figure(figsize=(8, 5))
    for k in range(len(times)):
        plt.plot(x, thetas[k, :], label=f"t = {times[k]:.0f} s")
    plt.xlabel("Distance x [m]")
    plt.ylabel("Concentration θ [µg/m³]")
    plt.title(f"{title_prefix} pollutant concentration profiles")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    if show:
        plt.show()
    else:
        plt.close()


def animate_solution(x, times, thetas, title="Animation",
                     save_path=None, show=False):
    """
    Creates an animation of theta(x,t) over stored times.
    If save_path is given, saves as GIF/MP4 depending on extension.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    line, = ax.plot([], [], lw=2)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(0, np.max(thetas) * 1.1)
    ax.set_xlabel("Distance x [m]")
    ax.set_ylabel("Concentration θ [µg/m³]")
    ax.set_title(title)
    ax.grid(True)

    def update(frame):
        line.set_data(x, thetas[frame, :])
        ax.set_title(f"{title} — t = {times[frame]:.0f} s")
        return line,

    anim = FuncAnimation(fig, update, frames=len(times),
                         interval=300, blit=True)

    if save_path is not None:
        # Choose writer based on file extension
        ext = os.path.splitext(save_path)[1].lower()
        try:
            if ext in [".gif"]:
                anim.save(save_path, writer="pillow")
            else:
                anim.save(save_path, writer="ffmpeg")
            print(f"Saved animation to {save_path}")
        except Exception as e:
            print(f"Could not save animation ({save_path}): {e}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return anim


# ============================================================
# MAIN SCRIPT (RUN ALL TESTS)
# ============================================================

def main():
    # --------------------------------------------------------
    # BUILD BASE GRID
    # --------------------------------------------------------
    x, t, Nx, Nt = build_grid(L, T_end, dx, dt)

    # ============================================================
    # TEST CASE 1: CONSTANT U, SPIKE AT INLET
    # ============================================================
    theta0_1 = np.zeros(Nx, dtype=float)
    theta0_1[0] = 250.0   # 250 µg/m³ at x=0, 0 elsewhere

    def bc_test1(time):
        return 250.0       # constant inlet

    times1, thetas1 = run_model(
        x, t, theta0_1, U_base, bc_test1, store_every=STORE_EVERY
    )
    plot_profiles(
        x, times1, thetas1,
        title_prefix="Test 1 – constant U",
        save_path=os.path.join(OUTPUT_DIR, "test1_profiles.png")
    )
    animate_solution(
        x, times1, thetas1,
        title="Test 1 – Animation",
        save_path=os.path.join(OUTPUT_DIR, "test1_animation.gif")
    )
    print("Finished Test 1")

    # ============================================================
    # TEST CASE 2: INITIAL CONDITIONS FROM CSV
    # ============================================================
    theta0_2 = read_and_interpolate_ic(IC_FILENAME, x)

    # keep inlet equal to interpolated initial value at x=0
    theta_inlet0 = float(theta0_2[0])

    def bc_test2(time):
        return theta_inlet0

    times2, thetas2 = run_model(
        x, t, theta0_2, U_base, bc_test2, store_every=STORE_EVERY
    )
    plot_profiles(
        x, times2, thetas2,
        title_prefix="Test 2 – IC from CSV",
        save_path=os.path.join(OUTPUT_DIR, "test2_profiles.png")
    )
    animate_solution(
        x, times2, thetas2,
        title="Test 2 – Animation",
        save_path=os.path.join(OUTPUT_DIR, "test2_animation.gif")
    )
    print("Finished Test 2")

    # ============================================================
    # TEST CASE 3: PARAMETER SENSITIVITY (U, dt, dx)
    # ============================================================

    # ---- Sensitivity to U (velocity) ----
    U_values = [0.05, 0.1, 0.2]   # half, base, double

    plt.figure(figsize=(8, 5))
    for U_val in U_values:
        xU, tU, NxU, NtU = build_grid(L, T_end, dx, dt)
        theta0 = np.zeros(NxU, dtype=float)
        theta0[0] = 250.0

        def bc_U(time):
            return 250.0

        # store only initial & final
        timesU, thetasU = run_model(
            xU, tU, theta0, U_val, bc_U, store_every=NtU - 1
        )
        plt.plot(xU, thetasU[-1, :], label=f"U = {U_val:.2f} m/s")

    plt.xlabel("Distance x [m]")
    plt.ylabel("Concentration θ [µg/m³]")
    plt.title("Test 3 – Sensitivity to velocity U (final time)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "test3_sensitivity_U.png"), dpi=300)
    plt.close()

    # ---- Sensitivity to dt (temporal resolution) ----
    dt_values = [5.0, 10.0, 20.0]

    plt.figure(figsize=(8, 5))
    for dt_val in dt_values:
        x_dt, t_dt, Nx_dt, Nt_dt = build_grid(L, T_end, dx, dt_val)
        theta0 = np.zeros(Nx_dt, dtype=float)
        theta0[0] = 250.0

        def bc_dt(time):
            return 250.0

        times_dt, thetas_dt = run_model(
            x_dt, t_dt, theta0, U_base, bc_dt, store_every=Nt_dt - 1
        )
        plt.plot(x_dt, thetas_dt[-1, :], label=f"dt = {dt_val:.1f} s")

    plt.xlabel("Distance x [m]")
    plt.ylabel("Concentration θ [µg/m³]")
    plt.title("Test 3 – Sensitivity to time step dt (final time)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "test3_sensitivity_dt.png"), dpi=300)
    plt.close()

    # ---- Sensitivity to dx (spatial resolution) ----
    dx_values = [0.4, 0.2, 0.1]

    plt.figure(figsize=(8, 5))
    for dx_val in dx_values:
        x_dx, t_dx, Nx_dx, Nt_dx = build_grid(L, T_end, dx_val, dt)
        theta0 = np.zeros(Nx_dx, dtype=float)
        theta0[0] = 250.0

        def bc_dx(time):
            return 250.0

        times_dx, thetas_dx = run_model(
            x_dx, t_dx, theta0, U_base, bc_dx, store_every=Nt_dx - 1
        )
        plt.plot(x_dx, thetas_dx[-1, :], label=f"dx = {dx_val:.2f} m")

    plt.xlabel("Distance x [m]")
    plt.ylabel("Concentration θ [µg/m³]")
    plt.title("Test 3 – Sensitivity to spatial step dx (final time)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "test3_sensitivity_dx.png"), dpi=300)
    plt.close()

    print("Finished Test 3")

    # ============================================================
    # TEST CASE 4: EXPONENTIALLY DECAYING INLET
    # ============================================================
    x4, t4, Nx4, Nt4 = build_grid(L, T_end, dx, dt)
    theta0_4 = np.zeros(Nx4, dtype=float)  # initially clean river

    tau = 100.0  # decay time scale [s]

    def bc_test4(time):
        # source concentration decays exponentially from 250
        return 250.0 * np.exp(-time / tau)

    times4, thetas4 = run_model(
        x4, t4, theta0_4, U_base, bc_test4, store_every=STORE_EVERY
    )
    plot_profiles(
        x4, times4, thetas4,
        title_prefix="Test 4 – exponentially decaying inlet",
        save_path=os.path.join(OUTPUT_DIR, "test4_profiles.png")
    )
    print("Finished Test 4")

    # ============================================================
    # TEST CASE 5: VARIABLE VELOCITY PROFILE (±10% RANDOM)
    # ============================================================

    # space-varying velocity profile: ±10% perturbation of U_base
    rng = np.random.default_rng(123)
    u_profile = U_base * (1.0 + 0.1 * (2.0 * rng.random(size=Nx) - 1.0))

    theta0_5 = np.zeros(Nx, dtype=float)
    theta0_5[0] = 250.0

    def bc_test5(time):
        return 250.0

    times5_var, thetas5_var = run_model(
        x, t, theta0_5, u_profile, bc_test5, store_every=STORE_EVERY
    )
    times5_const, thetas5_const = run_model(
        x, t, theta0_5, U_base, bc_test5, store_every=STORE_EVERY
    )

    # compare final profiles
    plt.figure(figsize=(8, 5))
    plt.plot(x, thetas5_const[-1, :], label="constant U")
    plt.plot(x, thetas5_var[-1, :],   label="variable U(x)")
    plt.xlabel("Distance x [m]")
    plt.ylabel("Concentration θ [µg/m³]")
    plt.title("Test 5 – effect of variable velocity profile (final time)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "test5_variable_U.png"), dpi=300)
    plt.close()

    print("Finished Test 5")
    print("All test cases (1–5) finished.")


if __name__ == "__main__":
    main()
