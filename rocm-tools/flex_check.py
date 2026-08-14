"""Discriminate flex SAP overcounting vs simulation divergence on cloth."""
import mujoco
import numpy as np
import warp as wp
import mujoco_warp as mjw

XML = "/matx/u/knatalia/.bench_assets/cloth/scene.xml"

mjm = mujoco.MjModel.from_xml_path(XML)
mjd = mujoco.MjData(mjm)
mujoco.mj_forward(mjm, mjd)


def run(device, nworld, nsteps, report_every):
    with wp.ScopedDevice(device):
        m = mjw.put_model(mjm)
        d = mjw.put_data(mjm, mjd, nworld=nworld, nconmax=40000, njmax=60000)
        for i in range(nsteps):
            mjw.step(m, d)
            if (i + 1) % report_every == 0:
                wp.synchronize()
                nacon = int(d.nacon.numpy()[0]) if hasattr(d, "nacon") else -1
                qpos = d.qpos.numpy()
                qvel = d.qvel.numpy()
                print(
                    f"{device} step {i + 1}: nacon={nacon} "
                    f"max|qpos|={np.abs(qpos).max():.3f} max|qvel|={np.abs(qvel).max():.3f} "
                    f"nan={np.isnan(qpos).any() or np.isnan(qvel).any()}"
                )


print("=== GPU (cuda:0) 1000 steps ===")
run("cuda:0", 1, 1000, 100)
print("=== CPU reference 300 steps ===")
run("cpu", 1, 300, 100)
print("FLEX_CHECK_DONE")
