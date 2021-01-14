from manim import *
import itertools as it
import random


violet = "#bc80bd"
blue = "#80b1d3"
cream = "#ece6e2"


class NeuralNetwork(Scene):
    arguments = {
        "network_size": 1,
        "network_position": 3*RIGHT,
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
        self.display_code()

    def display_code(self):
        code_text = Code(r"code_references\3_code_reference.py")
        code_text.scale(0.5)
        code_text.shift(3*LEFT)

        t1 = Tex(" ... ")
        t1.next_to(code_text, UP)

        t2 = Tex(" ... ")
        t2.next_to(code_text, DOWN)

        self.play(Write(t1))
        self.play(Write(code_text), run_time=5)
        self.play(Write(t2))
        self.wait(17)

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

    def wiggle_animation(self):
        edges = VGroup(*it.chain(*self.edge_groups))
        self.play(LaggedStartMap(
            ApplyFunction, edges,
            lambda mob: (lambda m: m.rotate_in_place(np.pi/12).set_color(YELLOW), mob),
            rate_func=wiggle))

    def continual_animation(self):
        args = {
            "colors": [blue, blue, violet, violet],
            "n_cycles": 5,
            "max_width": 3,
            "exp_width": 7
        }
        self.internal_time = 1
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
        edge_group.set_stroke(red, 4)

        for i in range(len(self.layers)-1):
            self.layers[i].neurons.set_fill(color=NeuralNetwork.arguments["neuron_color"], opacity=1)
            self.play(FadeIn(self.layers[i].neurons))
            self.play(LaggedStartMap(ShowCreationThenDestruction, edge_group[i]))

        self.layers[-1].neurons.set_fill(color=NeuralNetwork.arguments["neuron_color"], opacity=1)
        self.play(FadeIn(self.layers[-1].neurons))






