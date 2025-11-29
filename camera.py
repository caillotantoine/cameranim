import numpy as np
from enum import Enum
from typing import Tuple, List
from scipy.spatial.transform import Rotation as R
import cv2
from PIL import Image
from tqdm import tqdm

class CameraType(Enum):
    PERSPECTIVE = 0
    EQUIDISTANT = 1
    EQUIRECTANGULAR = 2

def cart2sph(x: float, y:float, z:float, debug=False) -> Tuple[float, float, float]:
    """Convert a cartesian coordinate to polar coordinate in right hand, z being depth, y toward bottom of image.

    Args:
        x (float): cartesian x toward right
        y (float): cartesian y toward bottom
        z (float): cartesian z depth

    Returns:
        Tuple[float, float, float]: azimuth, incidence (angle from optical axis), distance
    """

    dist = np.sqrt(x*x + y*y + z*z)
    az = np.atan2(y, x)
    planDist = np.sqrt(x*x + y*y)
    # inc = np.asin(planDist/dist)
    inc = np.atan2(planDist, z)
    return (az, inc, dist)

class Camera:

    def __init__(self, width:int, height:int, fov:float, imgCircSize:float, camType:CameraType):
        self.fov = np.deg2rad(fov)
        self.width = int(width)
        self.height = int(height)
        self.camType = camType
        self.imgCircSize = imgCircSize
        self.f = 0.0
        self.cx = int(width / 2.0)
        self.cy = int(height / 2.0)

        self.internalT = np.array([[0, 0, 1, 0], [-1, 0, 0, 0, ], [0, -1, 0, 0], [0, 0, 0, 1]]).T

        if self.camType == CameraType.PERSPECTIVE:
            self.f = float(self.imgCircSize) / (2.0 * np.tan(self.fov / 2.0))
        elif self.camType == CameraType.EQUIDISTANT:
            self.f = self.imgCircSize / self.fov
        elif self.camType == CameraType.EQUIRECTANGULAR:
            if self.width != (2 * self.height):
                print(f"Error, equirectangular image size must be 2:1 ration ({self.width} ≠ 2x{self.height})")

    def getCenter(self) -> Tuple[int, int]:
        return (self.cx, self.cy)
    
    def getImgCircRadius(self) -> int:
        return int(self.imgCircSize / 2.0)
    
    def perspectiveProj(self, theta:float) -> float:
        return np.tan(theta) * self.f
    
    def equidistantProj(self, theta:float) -> float:
        return theta * self.f
    
    def pol2pix(self, az:float, r:float) -> Tuple[int, int]:
        x = r * np.cos(az) + self.cx
        y = r * np.sin(az) + self.cy
        return (int(x), int(y))
    
    def setPoseInWorld(self, translation:Tuple[float, float, float], rotation:Tuple[float, float, float]):
        self.pose = np.eye(4)
        tx, ty, tz = translation
        rx, ry, rz = rotation
        t = np.array([tx, ty, tz])
        rot = R.from_rotvec(np.array([rx, ry, rz])).as_matrix()
        self.pose[0:3, 0:3] = rot
        self.pose[0:3, 3] = t
        self.poseInv = np.linalg.inv(self.pose)

    def projectInCamFrame(self, pt:Tuple[float, float, float]) -> Tuple[float, float, float]:
        xin, yin, zin = pt
        ptInCam = self.internalT @ self.poseInv @ np.array([xin, yin, zin, 1.0])
        return (ptInCam[0], ptInCam[1], ptInCam[2])

    def projector(self, pt:Tuple[float, float, float], debug=False) -> Tuple[int, int]:
        ptInFrame = self.projectInCamFrame(pt)
        ptx, pty, ptz = ptInFrame
        
        if self.camType == CameraType.EQUIRECTANGULAR:
            # TODO
            # certainement des atan2 avec ptx et ptz pour l'axe x et pty et ptz pour l'axe y
            return
        
        az, inc, dist = cart2sph(ptx, pty, ptz, debug=debug)
        if debug:
            print(f"az: {az}, inc {inc}, dist {dist}, ptz {ptz}")
        if inc > (self.fov / 2.0):
            raise ValueError("Not visible")
        
        if self.camType == CameraType.PERSPECTIVE:
            r = self.perspectiveProj(inc)
        elif self.camType == CameraType.EQUIDISTANT:
            r = self.equidistantProj(inc)
        else:
            raise RuntimeError("Unknown camera model")
        
        return self.pol2pix(az, r), dist


class GifGenerator:

    def __init__(self) -> None:
        self.frames: List[Image.Image] = []

    def add(self, img, cvtBGR2RGB=True) -> None:
        if type(img) == Image.Image:
            self.frames.append(img)
        elif type(img) == np.ndarray:
            if cvtBGR2RGB:
                if len(img.shape) == 3: 
                    if img.shape[2] == 3: #BGR
                        self.frames.append(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
                    elif img.shape[2] == 4: #BGRA
                        self.frames.append(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)))
                    else:
                        raise RuntimeError(f"Unknown way to process the image as it has {img.shape[2]} channels")
                elif len(img.shape) == 2: #we consider it to be monochromatic, no convertion, we pass
                    pass
                else:
                    raise RuntimeError(f"Unknown image shape {img.shape}")
            else: #We consider it to be monochromatic, no convertion
                self.frames.append(Image.fromarray(img))
                    
    def buildImage(self, pathToSave:str = "image.gif", duration:float = 1.0/30.0*1000.0, loop:int = 0) -> None:
        self.frames[0].save(
            pathToSave,
            save_all=True,
            append_images=self.frames[1:],
            duration=duration, 
            loop=loop
        )

class Renderer:
    def __init__(self, cam:Camera):
        self.gif = GifGenerator()
        self.cam = cam
    
    def setStart(self, translation:Tuple[float, float, float], rotation:Tuple[float, float, float]):
        self.stx, self.sty, self.stz = translation
        self.srx, self.sry, self.srz = rotation

    def setEnd(self, translation:Tuple[float, float, float], rotation:Tuple[float, float, float]):
        self.etx, self.ety, self.etz = translation
        self.erx, self.ery, self.erz = rotation
    
    def render(self, pointcloud:Tuple[Tuple[float, float, float], Tuple[int, int, int]], ptSize=30, pathToSave:str = "image.gif", fps=1.0/30.0, duration=1.0):
        self.imgDuration = fps * 1000.0
        self.nstep = int(duration / fps)

        dtx = self.etx - self.stx
        dty = self.ety - self.sty
        dtz = self.etz - self.stz
        drx = self.erx - self.srx
        dry = self.ery - self.sry
        drz = self.erz - self.srz

        rtx = dtx/self.nstep
        rty = dty/self.nstep
        rtz = dtz/self.nstep
        rrx = drx/self.nstep
        rry = dry/self.nstep
        rrz = drz/self.nstep
      

        for step in tqdm(range(self.nstep)):
            #place the camera
            self.cam.setPoseInWorld((self.stx+rtx*step, self.sty+rty*step, self.stz+rtz*step),
                                    (self.srx+rrx*step, self.sry+rry*step, self.srz+rrz*step))
            
            # Set image transluscent
            img = np.zeros((cam_p.width, cam_p.height, 4), dtype=np.uint8)

            # ImageCircle in white
            cv2.circle(img, self.cam.getCenter(), self.cam.getImgCircRadius(), color=(255, 255, 255, 255), thickness=-1)
            circle_mask = img.copy()

            for pt, c in pointcloud: # draw each point
                try:
                    pointInImg, dist = self.cam.projector(pt)
                    cv2.circle(img, pointInImg, int(ptSize/np.sqrt(dist)), color=c, thickness=-1)
                except ValueError:
                    continue

            # Cleanup what is outside image circle
            mask = circle_mask[:,:,0] > 0
            img[mask, 3] = 255
            img[~mask, 3] = 0

            # circle around the image circle
            cv2.circle(img, self.cam.getCenter(), self.cam.getImgCircRadius()+2, color=(0, 0, 0, 255), thickness=3)

            # add the frame
            self.gif.add(img)
            

        # save the gif
        self.gif.buildImage(pathToSave=pathToSave, duration=self.imgDuration, loop=0)

if __name__ == "__main__":
    # Creation of a pointcloud
    pc = []

    print("Building Point Cloud...")
    for _ in tqdm(range(2000)):
        d = 0
        pos = (0, 0, 0)
        while d < 6.0**2:
            x = np.random.uniform(-100, 100)
            y = np.random.uniform(-100, 100)
            z = np.random.uniform(-100, 100)
            d = x*x + y*y + z*z
            pos = (x, y, z)

        color = ()
        for _ in range(3):
            color += (np.random.uniform(0, 200),)
        color += (255,)
        pc.append(((pos), (color)))

    pc.sort(key=lambda item: np.linalg.norm(item[0]), reverse=True)

    ptSize = 100
    dist = 3.0

    # Perspective camera
    cam_p = Camera(720, 720, 60, 712, CameraType.PERSPECTIVE)

    render = Renderer(cam_p)
    render.setStart((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    render.setEnd((dist, 0.0, 0.0), (0.0, 0.0, 0.0))
    render.render(pc, ptSize, "persp_translation_x.gif", duration=2)

    render = Renderer(cam_p)
    render.setStart((dist, 0.0, 0.0), (0.0, 0.0, 0.0))
    render.setEnd((dist, dist, 0.0), (0.0, 0.0, 0.0))
    render.render(pc, ptSize, "persp_translation_y.gif", duration=2)

    render = Renderer(cam_p)
    render.setStart((dist, 0.0, 0.0), (0.0, 0.0, 0.0))
    render.setEnd((dist, dist, 0.0), (0.0, 0.0, np.deg2rad(10)))
    render.render(pc, ptSize, "persp_rotation_z.gif", duration=2)

    # Equidistant camera
    cam_ed = Camera(720, 720, 210, 712, CameraType.EQUIDISTANT)

    pc = pc[::8]
    ptSize /= 1.0

    render = Renderer(cam_ed)
    render.setStart((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    render.setEnd((dist, 0.0, 0.0), (0.0, 0.0, 0.0))
    render.render(pc, ptSize, "equidist_translation_x.gif", duration=2)

    render = Renderer(cam_ed)
    render.setStart((dist, 0.0, 0.0), (0.0, 0.0, 0.0))
    render.setEnd((dist, dist, 0.0), (0.0, 0.0, 0.0))
    render.render(pc, ptSize, "equidist_translation_y.gif", duration=2)

    render = Renderer(cam_ed)
    render.setStart((dist, 0.0, 0.0), (0.0, 0.0, 0.0))
    render.setEnd((dist, dist, 0.0), (0.0, 0.0, np.deg2rad(10)))
    render.render(pc, ptSize, "equidist_rotation_z.gif", duration=2)

### GARBAGE:

# for z in [-2, 2]:
    #     for y in [-2, 2]:
    #         for x in range(-3, 4, 1):
    #             pc.append(((x, y, z), color1))

    # pc.append(((-2, -2, -1), color2))
    # pc.append(((2, 2, -1), color2))
    # pc.append(((-1, -2, 0), color2))
    # pc.append(((1, 2, 0), color2))
    # pc.append(((0, -2, 1), color2))
    # pc.append(((0, 2, 1), color2))
    # pc.append(((2, -2, -1), color2))
    # pc.append(((-2, 2, -1), color2))
    # pc.append(((1, -2, 0), color2))
    # pc.append(((-1, 2, 0), color2))

    # pc.append(((3, -1, -1), color3))
    # pc.append(((3, -1, 1), color3))
    # pc.append(((3, 1, -1), color3))
    # pc.append(((3, 1, 1), color3))
    # pc.append(((3, 0, 0), color3))
    # pc.append(((3, 0, 1), color4))
    # pc.append(((3, 0, -1), color4))
    # pc.append(((3, 1, 0), color4))
    # pc.append(((3, -1, 0), color4))

    # pc.append(((-3, -1, -1), color3))
    # pc.append(((-3, -1, 1), color3))
    # pc.append(((-3, 1, -1), color3))
    # pc.append(((-3, 1, 1), color3))
    # pc.append(((-3, 0, 0), color5))
    # pc.append(((-3, 0, 1), color5))
    # pc.append(((-3, 0, -1), color5))
    # pc.append(((-3, 1, 0), color5))
    # pc.append(((-3, -1, 0), color5))