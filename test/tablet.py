import pyglet


window = pyglet.window.Window(800, 600, "Whiteboard", resizable=True)

points = []

is_drawing = False
is_pen_drawing = False

tablets = pyglet.input.get_tablets()

if len(tablets) > 0:
    tablet = tablets[0]
    tablet = tablet.open(window)

    @tablet.event
    def on_motion(cursor, x, y, pressure, a, b):
        if pressure > 0:
            is_pen_drawing = True
            points.append((x, y))
        else:
            is_pen_drawing = True


@window.event
def on_mouse_press(x, y, button, modifiers):
    if button == pyglet.window.mouse.LEFT:
        global is_drawing

        is_drawing = True
    

@window.event
def on_mouse_release(x, y, button, modifiers):
    if button == pyglet.window.mouse.LEFT:
        global is_drawing

        is_drawing = False
        points.append(None)

@window.event
def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
    if is_drawing and not is_pen_drawing:
        pos = (x, y)
        if pos not in points:
            points.append(pos)

@window.event
def on_draw():
    window.clear()

    for i in range(0, len(points) - 1):
        if points[i] is not None and points[i + 1] is not None:
            pyglet.graphics.draw(2, pyglet.gl.GL_LINES, 
            ('v2f', points[i] + points[i + 1]))

pyglet.app.run()