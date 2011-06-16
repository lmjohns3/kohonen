#!/usr/bin/env python

# Copyright (c) 2009 Leif Johnson <leif@leifjohnson.net>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

'''A GTK-based test for Kohonen maps.

Press q to quit.

Press h to change the source data to a horseshoe shape.
Press s to change the source data to a square shape.
Press c to change the source data to a circle shape.

Press + and - to grow and shrink the source data.
Press the arrow keys to move the location of the source data.

Press space bar to toggle training.

Press 1, 2, etc. to toggle display of the first, second, etc. controller's
neurons and error bars.
'''

import cairo
import gobject
import gtk
import logging
import math
import numpy
import numpy.random as rng
import optparse
import sys

import lmj.kohonen

FLAGS = optparse.OptionParser()
FLAGS.add_option('-v', '--verbose', action='store_true', help='Be more verbose')
FLAGS.add_option('-z', '--scale', type=float, default=0,
                 help='Scale the initial distribution by this factor')
FLAGS.add_option('-x', '--x-offset', type=int, default=0,
                 help='Start the distribution at this horizontal offset')
FLAGS.add_option('-y', '--y-offset', type=int, default=0,
                 help='Start the distribution at this vertical offset')
FLAGS.add_option('-d', '--display', default='',
                 help='Hide the display for these offsets')
FLAGS.add_option('-s', '--source', default='square',
                 help='Start with this source distribution')
FLAGS.add_option('', '--smoothing', default=50, type=int,
                 help='Smooth errors at this rate')


def dot(ctx, center, color=(0, 0, 0), size=2, alpha=1.0):
    '''Draw a dot at the given location in the given color.'''
    ctx.set_source_rgba(*(color + (alpha, )))
    ctx.arc(center[0], center[1], size, 0, 2 * math.pi)
    ctx.fill()


class Controller(object):
    '''Combine a vector quantizer with code for displaying it graphically.'''

    def __init__(self, quantizer, color, source=None):
        self.quantizer = quantizer
        self.color = color
        self.source = source
        self.target = None

    def learn(self):
        self.target = self.source()
        self.quantizer.learn(self.target)

    def evaluate(self):
        return sum(self.quantizer.distances(self.source()).min()
                   for _ in xrange(10)) / 10

    def draw_neurons(self, ctx, viewer):
        for c in lmj.kohonen.itershape(self.quantizer.shape):
            size = 2
            if hasattr(self.quantizer, 'activity'):
                size = len(self.quantizer.activity) * self.quantizer.activity[c]
            dot(ctx, self.quantizer.neuron(c), self.color, size=size, alpha=0.75)
        if self.target is not None:
            dot(ctx, self.target)
        if len(self.quantizer.shape) == 2:
            ctx.set_source_rgba(*(self.color + (0.25, )))
            ctx.set_line_width(0.5)
            for x in xrange(0, self.quantizer.shape[0]):
                for y in xrange(0, self.quantizer.shape[1]):
                    if x > 0:
                        ctx.move_to(*self.quantizer.neuron((x - 1, y)))
                        ctx.line_to(*self.quantizer.neuron((x, y)))
                    if y > 0:
                        ctx.move_to(*self.quantizer.neuron((x, y - 1)))
                        ctx.line_to(*self.quantizer.neuron((x, y)))
            ctx.stroke()


class GrowingGasController(Controller):
    def draw_neurons(self, ctx, viewer):
        super(GrowingGasController, self).draw_neurons(ctx, viewer)

        ctx.set_source_rgba(*(self.color + (0.25, )))
        ctx.set_line_width(0.5)
        for a, b in zip(*numpy.where(self.quantizer._connections > -1)):
            if a > b:
                ctx.move_to(*self.quantizer.neuron(a))
                ctx.line_to(*self.quantizer.neuron(b))
        ctx.stroke()


class Source(object):
    c = numpy.zeros(2)
    e = 0

    @property
    def s(self):
        return 100 / (1 + numpy.exp(-self.e / 10.0))

    def translate(self, x, y):
        self.c += numpy.array([x, y])

    def scale(self, e):
        self.e += e

    def __call__(self):
        return self.s * self._sample() + self.c

    def draw(self, ctx):
        ctx.save()
        ctx.translate(*self.c)
        ctx.scale(self.s, self.s)
        ctx.set_source_rgb(0.7, 0.8, 0.9)
        self._draw(ctx)
        ctx.restore()


class SquareSource(Source):
    def _sample(self):
        return rng.uniform(-1, 1, 2)

    def _draw(self, ctx):
        ctx.move_to(-1, -1)
        ctx.line_to( 1, -1)
        ctx.line_to( 1,  1)
        ctx.line_to(-1,  1)
        ctx.line_to(-1, -1)
        ctx.fill()


class CircleSource(Source):
    def _sample(self):
        theta = 2 * rng.random() * math.pi
        return rng.random() * numpy.array([math.cos(theta), math.sin(theta)])

    def _draw(self, ctx):
        ctx.arc(0, 0, 1, 0, 2 * math.pi)
        ctx.fill()


class HorseshoeSource(Source):
    def _sample(self):
        theta = rng.random() * math.pi
        offset = 0.2
        if rng.random() < 0.3:
            theta *= -1
            offset *= -1
        r = rng.uniform(0.5, 1)
        return r * numpy.array([math.cos(theta), math.sin(theta) + offset])

    def _draw(self, ctx):
        for y, arc1, arc2 in ((0.2, ctx.arc, ctx.arc_negative),
                              (-0.2, ctx.arc_negative, ctx.arc)):
            ctx.move_to(0.5, y)
            ctx.line_to(1, y)
            arc1(0, y, 1, 0, math.pi)
            ctx.line_to(-0.5, y)
            arc2(0, y, 0.5, math.pi, 0)
            ctx.fill()


class Viewer(gtk.DrawingArea):
    __gsignals__ = {'expose-event': 'override'}

    def __init__(self, options, *controllers):
        super(Viewer, self).__init__()

        self.controllers = controllers
        self.errors = [[] for _ in self.controllers]

        self.smoothing = options.smoothing

        self.visible = [False] * 10
        for offset in getattr(options, 'display', ''):
            self.visible[int(offset)] = True

        self.teacher = None
        self.iteration = 0

        if options.source.startswith('c'):
            self.set_source(CircleSource())
        elif options.source.startswith('h'):
            self.set_source(HorseshoeSource())
        else:
            self.set_source(SquareSource())

        self.source.translate(options.x_offset, options.y_offset)
        self.source.scale(options.scale)

    def set_source(self, source):
        self.source = source
        for c in self.controllers:
            c.source = source

    def do_keypress(self, window, event):
        k = event.string or gtk.gdk.keyval_name(event.keyval).lower()

        if not k:
            return

        elif k == 'q':
            gtk.main_quit()

        elif k in '0123456789':
            self.visible[int(event.string)] ^= True

        elif k == ' ':
            if self.teacher:
                gobject.source_remove(self.teacher)
                self.teacher = None
            else:
                self.teacher = gobject.timeout_add(50, self.learn)

        elif k == 'r':
            self.learn()

        elif k == 'h':
            self.set_source(HorseshoeSource())
        elif k == 's':
            self.set_source(SquareSource())
        elif k == 'c':
            self.set_source(CircleSource())

        elif k in '+=':
            self.source.scale(5)
        elif k in '-':
            self.source.scale(-5)

        elif k == 'up':
            self.source.translate(0, 10)
        elif k == 'down':
            self.source.translate(0, -10)
        elif k == 'left':
            self.source.translate(-10, 0)
        elif k == 'right':
            self.source.translate(10, 0)

        self.queue_draw()

    def learn(self):
        for i, ctrl in enumerate(self.controllers):
            ctrl.learn()
            if len(self.errors[i]) >= 181 * self.smoothing:
                self.errors[i] = []
            self.errors[i].append(ctrl.evaluate())
        self.iteration += 1
        self.queue_draw()
        return True

    def iterctrl(self):
        for i, ctrl in enumerate(self.controllers):
            if self.visible[i+1]:
                yield ctrl, self.errors[i]

    def do_expose_event(self, event):
        ctx = self.window.cairo_create()
        ctx.rectangle(event.area.x,
                      event.area.y,
                      event.area.width,
                      event.area.height)
        ctx.clip()

        width, height = self.window.get_size()
        ctx.set_source_rgb(1, 1, 1)
        ctx.rectangle(0, 0, width, height)
        ctx.fill()

        ctx.set_source_rgb(0, 0, 0)
        ctx.move_to(10, 20)
        ctx.show_text('iteration %d' % self.iteration)

        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        ctx.translate(width / 2, height / 2)
        z = min(width / 200.0, height / 200.0)
        ctx.scale(z, -z)

        self.draw_neurons(ctx)
        self.draw_errors(ctx)

    def draw_neurons(self, ctx):
        ctx.save()
        ctx.translate(0, 20)
        if self.visible[0]:
            self.source.draw(ctx)
        for ctrl, _ in self.iterctrl():
            ctrl.draw_neurons(ctx, self)
        ctx.restore()

    def draw_errors(self, ctx):
        ctx.save()
        ctx.translate(-90, -90)
        ctx.set_source_rgba(0, 0, 0, 0.9)
        ctx.set_line_width(0.5)
        ctx.move_to(180, 0)
        ctx.line_to(0, 0)
        ctx.line_to(0, 10)
        ctx.stroke()

        for ctrl, err in self.iterctrl():
            ctx.set_source_rgba(*(ctrl.color + (0.75, )))
            ctx.set_line_width(1)

            total = end_x = end_y = 0
            for i, y in enumerate(err):
                total += y
                if (i + 1) % self.smoothing == 0:
                    x = i / self.smoothing
                    y = total / self.smoothing
                    total = 0
                    if x == 0:
                        ctx.move_to(x, y)
                    else:
                        ctx.line_to(x, y)
                    if len(err) - i < self.smoothing:
                        end_x = x
                        end_y = y

            ctx.stroke()

            if end_x:
                dot(ctx, (end_x, end_y), ctrl.color, alpha=0.75)
                ctx.move_to(end_x + 5, end_y)
                ctx.scale(1, -1)
                ctx.show_text('%.2f' % end_y)
                ctx.scale(1, -1)

        ctx.restore()


DARK_GRAY = (0.3, 0.3, 0.3)
RED = (0.7, 0.3, 0.3)
BLUE = (0.3, 0.3, 0.7)
GREEN = (0.3, 0.7, 0.3)
YELLOW = (0.7, 0.7, 0.3)


if __name__ == '__main__':
    opts, args = FLAGS.parse_args()

    logging.basicConfig(stream=sys.stdout,
                        level=opts.verbose and logging.DEBUG or logging.INFO,
                        format='%(levelname).1s %(asctime)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    ET = lmj.kohonen.ExponentialTimeseries

    def kwargs(shape=(8, 8), z=0.2):
        return dict(dimension=2,
                    shape=shape,
                    learning_rate=ET(-5e-4, 1, 0.1),
                    noise_variance=z)

    kw = kwargs()
    m = lmj.kohonen.Map(lmj.kohonen.Parameters(**kw))
    m.reset()

    kw = kwargs((64, ))
    g = lmj.kohonen.Gas(lmj.kohonen.Parameters(**kw))
    g.reset()

    kw = kwargs((64, ))
    kw['growth_interval'] = 7
    kw['max_connection_age'] = 17
    gg = lmj.kohonen.GrowingGas(lmj.kohonen.GrowingGasParameters(**kw))
    gg.reset()

    kw = kwargs()
    fm = lmj.kohonen.Filter(lmj.kohonen.Map(lmj.kohonen.Parameters(**kw)))
    fm.reset()

    kw = kwargs((64, ))
    fg = lmj.kohonen.Filter(lmj.kohonen.Gas(lmj.kohonen.Parameters(**kw)))
    fg.reset()

    v = Viewer(opts,
               Controller(m, DARK_GRAY),
               Controller(g, RED),
               GrowingGasController(gg, BLUE),
               Controller(fm, GREEN),
               Controller(fg, YELLOW),
               )
    v.show()

    w = gtk.Window()
    w.set_title('Kohonen')
    w.set_default_size(800, 600)
    w.add(v)
    w.present()
    w.connect('key-press-event', v.do_keypress)
    w.connect('delete-event', gtk.main_quit)

    gtk.main()
