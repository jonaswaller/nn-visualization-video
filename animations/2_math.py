from manim import *
import itertools as it
import random

red = "ffc1b6"
cream = "#ece6e2"


class NeuralNetwork(Scene):
    arguments = {
        "network_size": 1,
        "network_position": ORIGIN,
        "layer_sizes": [6, 7, 7, 6],
        "layer_buff": LARGE_BUFF,
        "neuron_radius": 0.15,
        "neuron_color": cream,
        "neuron_width": 3,
        "neuron_fill_color": BLACK,
        "neuron_fill_opacity": 1,
        "neuron_buff": MED_SMALL_BUFF,
        "edge_color": cream,
        "edge_width": 1.25,
        "edge_opacity": 0.75,
        "layer_label_color": WHITE,
        "layer_label_size": 0.5,
        "neuron_label_color": WHITE
    }

    def construct(self):
        self.add_neurons()
        self.edge_security()  # turn on for continual_animation
        #self.add_edges()  # turn off for continual_animation
        #self.label_layers()
        #self.label_neurons()
        #self.pulse_animation()
        #self.pulse_animation_2()
        #self.wiggle_animation()
        self.continual_animation()
        #self.forward_pass_animation()
        self.math_text()

    def math_text(self):
        title1 = Tex("Calculus")
        title1.set_color(GOLD_C)
        title1.shift(3*UP + 4.5*LEFT)
        title1.scale(0.75)

        t1 = MathTex(r"f'(x)=lim_{h \to 0}\frac{f(x+h)-f(x)}{h}")
        t1.shift(2*UP + 4.5*LEFT)
        t1.scale(0.5)

        t2 = MathTex(r"F(x)=f(x_0)+\int_{x_0}^{x}f(x)dx")
        t2.shift(UP + 4.5*LEFT)
        t2.scale(0.5)

        t3 = MathTex(r"\frac{\partial f}{\partial x}=lim_{h \to 0}\frac{f(x+h,y)-f(x,y)}{h}")
        t3.shift(4.5*LEFT)
        t3.scale(0.5)

        t4 = MathTex(r"\frac{\partial f}{\partial y}=lim_{h \to 0}\frac{f(x+h,y)-f(x,y)}{h}")
        t4.shift(DOWN + 4.5*LEFT)
        t4.scale(0.5)

        t5 = MathTex(r"\nabla f(x_1,x_2,...,x_n)=(\frac{\partial f}{\partial x_1},\frac{\partial f}{\partial x_2},...,\frac{\partial f}{\partial x_n})")
        t5.shift(2*DOWN + 4.5*LEFT)
        t5.scale(0.5)

        t6 = MathTex(r"min_{w}l(N(x_{train},w))")
        t6.shift(3*DOWN + 4.5*LEFT)
        t6.scale(0.5)

        title2 = Tex("Linear Algebra")
        title2.set_color(GOLD_C)
        title2.shift(3*UP + 4.5*RIGHT)
        title2.scale(0.75)

        t7 = MathTex(r"(x_1,x_2)+(y_1,y_2)=(x_1+y_1,x_2+y_2)")
        t7.shift(2*UP + 4.5*RIGHT)
        t7.scale(0.5)

        t8 = MathTex(r"\left\lVert x \right\rVert_{p}=(\sum_{i-1}^{n}(x_i)^p)^{1/p}")
        t8.shift(UP + 4.5*RIGHT)
        t8.scale(0.5)

        t9 = MathTex(r"V=\left \{ \sum_{i=1}^{n}\alpha_i b_i:\alpha_i \in \mathbb{R} \right \}")
        t9.shift(4.5*RIGHT)
        t9.scale(0.5)

        t10 = MathTex(r"f(x)=\sigma(Ax+b)")
        t10.shift(DOWN + 4.5*RIGHT)
        t10.scale(0.5)

        t11 = MathTex(r"L(ax+y)=aL(x)+L(y)")
        t11.shift(2*DOWN + 4.5*RIGHT)
        t11.scale(0.5)

        t12 = MathTex(r"AB=(\sum_{k=1}^{m}a_{i,k}b_{k,j})^{n,l}_{i,j=1}")
        t12.shift(3*DOWN + 4.5*RIGHT)
        t12.scale(0.5)

        self.play(Write(title1), Write(title2))
        self.play(Write(t1), Write(t7))
        self.play(Write(t2), Write(t8))
        self.play(Write(t3), Write(t9))
        self.play(Write(t4), Write(t10))
        self.play(Write(t5), Write(t11))
        self.play(Write(t6), Write(t12))
        self.wait(2)

    def add_neurons(self):
        layers = VGroup(*[self.get_layer(size) for size in NeuralNetwork.arguments["layer_sizes"]])
        layers.arrange(RIGHT, buff=NeuralNetwork.arguments["layer_buff"])
        layers.scale(NeuralNetwork.arguments["network_size"])
        # self.layers is layers, but we can use it throughout every method in our class
        # without having to redefine layers each time
        self.layers = layers
        layers.shift(NeuralNetwork.arguments["network_position"])
        self.play(Write(layers))

    def get_layer(self, size):
        layer = VGroup()
        n_neurons = size
        neurons = VGroup(*[
            Circle(
                radius=NeuralNetwork.arguments["neuron_radius"],
                stroke_color=NeuralNetwork.arguments["neuron_color"],
                stroke_width=NeuralNetwork.arguments["neuron_width"],
                fill_color=NeuralNetwork.arguments["neuron_fill_color"],
                fill_opacity=NeuralNetwork.arguments["neuron_fill_opacity"],
            )
            for i in range(n_neurons)
        ])
        neurons.arrange(DOWN, buff=NeuralNetwork.arguments["neuron_buff"])
        layer.neurons = neurons
        layer.add(neurons)
        return layer

    def edge_security(self):
        self.edge_groups = VGroup()
        for l1, l2 in zip(self.layers[:-1], self.layers[1:]):
            edge_group = VGroup()
            for n1, n2 in it.product(l1.neurons, l2.neurons):
                edge = self.get_edge(n1, n2)
                edge_group.add(edge)
            self.edge_groups.add(edge_group)

    def add_edges(self):
        self.edge_groups = VGroup()
        for l1, l2 in zip(self.layers[:-1], self.layers[1:]):
            edge_group = VGroup()
            for n1, n2 in it.product(l1.neurons, l2.neurons):
                edge = self.get_edge(n1, n2)
                edge_group.add(edge)
            self.play(Write(edge_group), run_time=0.5)
            self.edge_groups.add(edge_group)

    def get_edge(self, neuron1, neuron2):
        return Line(
            neuron1.get_center(),
            neuron2.get_center(),
            buff=1.25*NeuralNetwork.arguments["neuron_radius"],
            stroke_color=NeuralNetwork.arguments["edge_color"],
            stroke_width=NeuralNetwork.arguments["edge_width"],
            stroke_opacity=NeuralNetwork.arguments["edge_opacity"]
        )

    def label_layers(self):
        input_layer = VGroup(*self.layers[0])
        input_label = Tex("Input Layer")
        input_label.set_color(NeuralNetwork.arguments["layer_label_color"])
        input_label.scale(NeuralNetwork.arguments["layer_label_size"])
        input_label.next_to(input_layer, UP, SMALL_BUFF)
        self.play(Write(input_label))

        hidden_layer = VGroup(*self.layers[1:3])
        hidden_label = Tex("Hidden Layers")
        hidden_label.set_color(NeuralNetwork.arguments["layer_label_color"])
        hidden_label.scale(NeuralNetwork.arguments["layer_label_size"])
        hidden_label.next_to(hidden_layer, UP, SMALL_BUFF)
        self.play(Write(hidden_label))

        output_layer = VGroup(*self.layers[-1])
        output_label = Tex("Output Layer")
        output_label.set_color(NeuralNetwork.arguments["layer_label_color"])
        output_label.scale(NeuralNetwork.arguments["layer_label_size"])
        output_label.next_to(output_layer, UP, SMALL_BUFF)
        self.play(Write(output_label))

    def label_neurons(self):
        input_labels = VGroup()
        for n, neuron in enumerate(self.layers[0].neurons):
            label = MathTex(f"x_{n + 1}")
            label.set_height(0.3 * neuron.get_height())
            label.set_color(NeuralNetwork.arguments["neuron_label_color"])
            label.move_to(neuron)
            input_labels.add(label)
        self.play(Write(input_labels))

        weight_labels = VGroup()
        for n, neuron in enumerate(self.layers[2].neurons):
            label = MathTex(f"w_{n + 1}")
            label.set_height(0.3 * neuron.get_height())
            label.set_color(NeuralNetwork.arguments["neuron_label_color"])
            label.move_to(neuron)
            weight_labels.add(label)
        self.play(Write(weight_labels))

        output_labels = VGroup()
        for n, neuron in enumerate(self.layers[-1].neurons):
            label = MathTex(r"\hat{y}_" + "{" + f"{n + 1}" + "}")
            label.set_height(0.4 * neuron.get_height())
            label.set_color(NeuralNetwork.arguments["neuron_label_color"])
            label.move_to(neuron)
            output_labels.add(label)
        self.play(Write(output_labels))

    def pulse_animation(self):
        edge_group = self.edge_groups.copy()
        edge_group.set_stroke(YELLOW, 4)  # color, width
        for i in range(3):
            self.play(LaggedStartMap(
                ShowCreationThenDestruction, edge_group,
                run_time=1.5))
            self.wait()

    def pulse_animation_2(self):
        edge_group = VGroup(*it.chain(*self.edge_groups))
        edge_group = edge_group.copy()
        edge_group.set_stroke(YELLOW, 4)  # color, width
        for i in range(3):
            self.play(LaggedStartMap(
                ShowCreationThenDestruction, edge_group,
                run_time=1.5))
            self.wait()

    def wiggle_animation(self):
        edges = VGroup(*it.chain(*self.edge_groups))
        self.play(LaggedStartMap(
            ApplyFunction, edges,
            lambda mob: (lambda m: m.rotate_in_place(np.pi/12).set_color(YELLOW), mob),
            rate_func=wiggle))

    def continual_animation(self):
        args = {
            "colors": [cream, cream, red, red],
            "n_cycles": 5,
            "max_width": 3,
            "exp_width": 7
        }
        self.internal_time = 1  # no initial lag :)
        self.move_to_targets = []
        edges = VGroup(*it.chain(*self.edge_groups))
        for edge in edges:
            edge.colors = [random.choice(args["colors"]) for i in range(args["n_cycles"])]
            msw = args["max_width"]
            edge.widths = [msw * random.random()**args["exp_width"] for i in range(args["n_cycles"])]
            edge.cycle_time = 1 + random.random()

            edge.generate_target()
            edge.target.set_stroke(edge.colors[0], edge.widths[0])
            edge.become(edge.target)
            self.move_to_targets.append(edge)

        self.edges = edges
        animation = self.edges.add_updater(lambda m, dt: self.update_edges(dt))
        self.play(FadeIn(animation), run_time=0.05)

    def update_edges(self, dt):
        self.internal_time += dt
        if self.internal_time < 1:
            alpha = smooth(self.internal_time)
            for i in self.move_to_targets:
                i.update(alpha)
            return
        for edge in self.edges:
            t = (self.internal_time-1)/edge.cycle_time
            alpha = ((self.internal_time-1)%edge.cycle_time)/edge.cycle_time
            low_n = int(t)%len(edge.colors)
            high_n = int(t+1)%len(edge.colors)
            color = interpolate_color(edge.colors[low_n], edge.colors[high_n], alpha)
            width = interpolate(edge.widths[low_n], edge.widths[high_n], alpha)
            edge.set_stroke(color, width)

    def forward_pass_animation(self):
        edge_group = self.edge_groups.copy()
        edge_group.set_stroke(red, 4)  # c1: red

        for i in range(len(self.layers)-1):
            self.layers[i].neurons.set_fill(color=NeuralNetwork.arguments["neuron_color"], opacity=1)
            self.play(FadeIn(self.layers[i].neurons))
            self.play(LaggedStartMap(ShowCreationThenDestruction, edge_group[i]))

        self.layers[-1].neurons.set_fill(color=NeuralNetwork.arguments["neuron_color"], opacity=1)
        self.play(FadeIn(self.layers[-1].neurons))






