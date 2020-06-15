"""Observer classes."""
import glob
import os
import queue
import sys
try:
    sys.path.append(glob.glob('/opt/carla-simulator/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla
import numpy as np

CARLA_UAV_SCALE = 20.
CARLA_UAV_HEIGHT = 40.
CARLA_CAR_SCALE = 11.
CARLA_CAR_HEIGHT = 3.
CARLA_IM_WIDTH = 400
CARLA_IM_HEIGHT = 300

class CarlaObservations():
    """CARLA observer object.

    Parameters
    ----------
    host : str
        IP address of server running CARLA.
    port : int
        Port for CARLA.
    setting : str
        Observation setting, either uav or car.

    """
    def __init__(self, host='192.168.1.38', port=2000,
                 setting='uav'):
        self.client = carla.Client(host, port)
        self.client.set_timeout(2.0)
        self.world = self.client.get_world()

        # Setting up world
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        self.world.apply_settings(settings)

        weather = self.world.get_weather()
        weather.cloudiness = 0.
        weather.wind_intensity = 0.
        weather.precipitation = 0.
        weather.fog_density = 0.
        weather.wetness = 0.
        weather.sun_azimuth_angle = 0.
        weather.sun_altitude_angle = 90.
        self.world.set_weather(weather)
    
        # Set up initial location and scaling
        self.setting = setting
        if self.setting == 'uav':
            self.z_pos = CARLA_UAV_HEIGHT
            self.pitch = -90
            self.scale = CARLA_UAV_SCALE
        elif self.setting == 'car':
            self.z_pos = CARLA_CAR_HEIGHT
            self.pitch = 0
            self.scale = CARLA_CAR_SCALE

        self.actor_list = []
        self.spawn_camera()

        ndims = 2
        self.C = np.zeros([ndims, ndims*2])
        for i in range(ndims):
            self.C[i, 2*i] = 1.

    def step(self):
        """Advance simulator by a step."""
        self.world.tick()
        image = self.image_queue.get()
        self.images.append(image)

    def spawn_camera(self,):
        """Initialize camera."""
        blueprint_library = self.world.get_blueprint_library()
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '{}'.format(CARLA_IM_WIDTH))
        camera_bp.set_attribute('image_size_y', '{}'.format(CARLA_IM_HEIGHT))

        camera_transform = carla.Transform(carla.Location(x=0., z=self.z_pos),
                                           carla.Rotation(pitch=self.pitch))
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=None)
        self.camera.set_simulate_physics(enabled=False)
        self.camera.set_transform(camera_transform)
        self.actor_list.append(self.camera)
        print('created %s' % self.camera.type_id)
        self.image_queue = queue.Queue()
        self.camera.listen(self.image_queue.put)
        self.images = []

    def observe(self, x, angle_offset=0.):
        """Get observations from given pos/angle."""
        xpos, ypos = self.C @ x

        location = self.camera.get_location()
        location.x = self.scale * xpos
        location.y = self.scale * ypos
        location.z = self.z_pos

        if self.setting == 'uav':
            angle = 0. + angle_offset
        elif self.setting == 'car':
            vx = x[1]
            vy = x[3]
            angle = np.degrees(np.arctan2(vy, vx)) + angle_offset

        camera_transform = carla.Transform(location,
                                           carla.Rotation(yaw=angle,
                                                          pitch=self.pitch))
        self.camera.set_transform(camera_transform)
        self.step()
        return self.im_to_array(self.images[-1])

    def im_to_array(self, image):
        """Read image as array."""
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        return array[:, :, [2, 1, 0]]

    def kill(self):
        """Destroy actors."""
        print('destroying actors')
        self.camera.destroy()
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
        print('done.')

class DotObservations():
    def __init__(self, ndims=2, im_size=100, im_scale=5., dot_width=0.05, sv=0):
        assert ndims <= 3, "dot only operates in <=3 dim"
        self.ndims = ndims
        self.C = np.zeros([ndims, ndims*2])
        for i in range(ndims):
            self.C[i, 2*i] = 1.
        self.noise_level = sv
        self.n = im_size
        self.width = dot_width
        self.scale = im_scale
    def observe(self, x):
        pos = self.C @ x / self.scale
        if self.ndims == 1:
            loc = (pos[0], 0)
            width = self.width
        elif self.ndims == 2:
            loc = pos
            width = self.width
        elif self.ndims == 3:
            loc = pos[0:2]
            height_multiplier = 1/(1+pos[2])
            width = self.width * height_multiplier
        im = loc2im(loc, self.n, width, self.noise_level, circular=False)
        return im

class LinearObservations():
    def __init__(self, C, sv=0):
        self.C = C
        self.sv = sv

    def observe(self, x):
        y = self.C @ x
        noise = np.clip(np.random.normal(size=y.shape, scale=self.sv), -1, 1)
        return y + noise

class PolynomialObservations():
    def __init__(self, pows, sv=0, q=100, q_lim=20):
        self.C = np.array([[1, 0]])
        self.Bv = sv * np.eye(self.C.shape[0])
        self.G = np.random.normal(size=[q, len(pows)*2*q_lim])
        self.indicator_powers = lambda y: [np.power(indicator_observe(y, lim=(-q_lim, q_lim)), 
                                                    1/p) for p in pows]
    def observe(self, x):
        y = self.C @ x + self.Bv @ np.random.normal(scale=1, size=self.Bv.shape[1])
        return self.G @ np.hstack(self.indicator_powers(y))

class IndicatorObservations():
    def __init__(self, im_size=100, im_scale=5., sv=0):
        self.C = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
        self.noise_level = sv
        self.n = im_size
        self.scale = im_scale
    def observe(self, x):
        loc = self.C @ x / self.scale * self.n
        im = indicator_observe(loc, lim=(-self.n//2, self.n//2))
        return im

## Static observation functions

def loc2im(loc, n, width, noise_level, circular=False):
    # from https://github.com/modestyachts/robust-control-from-vision
    x, y = loc
    if circular == 'y':
        y = (y + 0.5) % 1 - 0.5
    elif circular == 'x':
        x = (x + 0.5) % 1 - 0.5
    yy, xx = np.mgrid[:n, :n]/n - 0.5
    im = np.exp(-0.5*((xx - x)**2 + (yy - y)**2) / width**2)
    im += np.random.standard_normal((n, n)) * noise_level * im.max()
    if (im.max() - im.min()) > 0:
        return (im - im.min())/(im.max() - im.min())
    return im

def custom_pth_root(y, p):
    assert len(y.shape) < 2, "method written for vectors"
    assert p % 2 == 1, "must be odd root"
    y_pow = np.zeros_like(y)
    for i in range(y.shape[0]):
        y_pow[i] = np.abs(y[i])**(1/p) * np.sign(y[i])
    return y_pow

def indicator_observe(y, lim=(-1000, 1000)):
    assert len(y.shape) < 2, "method written for vectors"
    z_dims = []
    for i in range(y.shape[0]):
        z_i = np.zeros(lim[1]-lim[0])
        if y[i] < lim[1]-1 and y[i] >= lim[0]:
            ind0 = int(np.floor(y[i])-lim[0])
            z_i[ind0] = 1 - (y[i] - np.floor(y[i]))
            try:
                z_i[ind0+1] = y[i] - np.floor(y[i])
            except IndexError as e:
                print(y[i], lim, ind0, ind0+1)
        z_dims.append(z_i)
    return np.outer(*z_dims)

def basis_functions(z, pows):
    return np.hstack([np.power(z, p) for p in pows])

class IndexTracker():
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        self.X = X
        self.slices = len(X)
        self.ind = 0

        self.im = ax.imshow(self.X[self.ind])
        self.update()

    def onscroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind - 1) % self.slices
        else:
            self.ind = (self.ind + 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[self.ind])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()
