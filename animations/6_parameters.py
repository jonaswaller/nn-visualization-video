from manim import *
import itertools as it
import random


red = "ffc1b6"
brown = "fad586"


class NeuralNetwork(Scene):
    arguments = {
        "network_size": 1,
        "network_position": 3*LEFT,
        "layer_sizes": [4, 5, 5, 4],
        "layer_buff": LARGE_BUFF,
        "neuron_radius": 0.15,
        "neuron_color": TEAL,
        "neuron_width": 3,
        "neuron_fill_color": BLACK,
        "neuron_fill_opacity": 1,
        "neuron_buff": MED_SMALL_BUFF,
        "edge_color": red,
        "edge_width": 1.25,
        "edge_opacity": 0.75,
        "layer_label_color": WHITE,
        "layer_label_size": 0.5,
        "neuron_label_color": WHITE
    }

    def construct(self):
        #self.add_neurons([6, 7, 7, 6])
        #self.edge_security()  # turn on for continual_animation
        #self.add_edges()  # turn off for continual_animation
        #self.label_layers()
        #self.label_neurons()
        #self.pulse_animation()
        #self.pulse_animation_2()
        #self.wiggle_animation()
        #self.continual_animation()
        #self.forward_pass_animation()
        self.show_parameters()

    def add_neurons(self, layer_size):
        layers = VGroup(*[self.get_layer(size) for size in layer_size])
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
            "colors": [WHITE, red, brown],
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

    def show_parameters(self):
        # FIRST ARCHITECTURE
        self.add_neurons([4, 5, 5, 4])
        self.edge_security()
        self.continual_animation()

        layer_sizes = Tex("Layer sizes:")
        layer_sizes.shift(UP + 2*RIGHT)
        arch_1 = Tex("4, 5, 5, 4")
        arch_1.set_color(TEAL)
        arch_1.shift(UP + 2*RIGHT)
        arch_1.next_to(layer_sizes, RIGHT)
        weights = Tex("Weights:")
        weights.next_to(layer_sizes, DOWN)
        biases = Tex("Biases:")
        biases.next_to(weights, DOWN)
        params = Tex("Parameters:")
        params.next_to(biases, DOWN)
        group = VGroup(layer_sizes, arch_1, weights, biases, params)
        self.play(Write(group))

        # weights
        weight_calc = Tex("(4x5)+(5x5)+(5x4)")
        weight_calc.scale(0.75)
        total_weights = Tex("65")
        total_weights.set_color(TEAL)
        weight_calc.next_to(weights, RIGHT)
        total_weights.next_to(weights, RIGHT)
        self.play(Write(weight_calc))
        self.play(ReplacementTransform(weight_calc, total_weights))

        # biases
        bias_calc = Tex("4+5+5+4")
        total_biases = Tex("18")
        total_biases.set_color(TEAL)
        bias_calc.next_to(biases, RIGHT)
        total_biases.next_to(biases, RIGHT)
        self.play(Write(bias_calc))
        self.play(ReplacementTransform(bias_calc, total_biases))

        # parameters
        param_calc = Tex("65+18")
        total_params = Tex("83")
        total_params.set_color(TEAL)
        param_calc.next_to(params, RIGHT)
        total_params.next_to(params, RIGHT)
        self.play(Write(param_calc))
        self.play(ReplacementTransform(param_calc, total_params))

        # network + stats go away
        network = VGroup(self.layers, self.edge_groups)
        calculations = VGroup(total_weights, total_biases, total_params)
        group = VGroup(network, calculations)
        self.play(FadeOut(group))

        # SECOND ARCHITECTURE
        arch_2 = Tex("6, 7, 7, 6")
        arch_2.set_color(TEAL)
        arch_2.shift(UP + 2*RIGHT)
        arch_2.next_to(layer_sizes, RIGHT)
        self.play(ReplacementTransform(arch_1, arch_2))
        self.add_neurons([6, 7, 7, 6])
        self.edge_security()
        self.continual_animation()

        # weights
        weight_calc = Tex("(6x7)+(7x7)+(7x6)")
        weight_calc.scale(0.75)
        total_weights = Tex("133")
        total_weights.set_color(TEAL)
        weight_calc.next_to(weights, RIGHT)
        total_weights.next_to(weights, RIGHT)
        self.play(Write(weight_calc))
        self.play(ReplacementTransform(weight_calc, total_weights))

        # biases
        bias_calc = Tex("6+7+7+6")
        total_biases = Tex("26")
        total_biases.set_color(TEAL)
        bias_calc.next_to(biases, RIGHT)
        total_biases.next_to(biases, RIGHT)
        self.play(Write(bias_calc))
        self.play(ReplacementTransform(bias_calc, total_biases))

        # parameters
        param_calc = Tex("133+26")
        total_params = Tex("159")
        total_params.set_color(TEAL)
        param_calc.next_to(params, RIGHT)
        total_params.next_to(params, RIGHT)
        self.play(Write(param_calc))
        self.play(ReplacementTransform(param_calc, total_params))

        # network + stats go away
        network = VGroup(self.layers, self.edge_groups)
        calculations = VGroup(total_weights, total_biases, total_params)
        group = VGroup(network, calculations)
        self.play(FadeOut(group))

        # THIRD ARCHITECTURE
        arch_3 = Tex("7, 9, 9, 5, 2")
        arch_3.set_color(TEAL)
        arch_3.shift(UP + 2*RIGHT)
        arch_3.next_to(layer_sizes, RIGHT)
        self.play(ReplacementTransform(arch_2, arch_3))
        self.add_neurons([7, 9, 9, 5, 2])
        self.edge_security()
        self.continual_animation()

        # weights
        weight_calc = Tex("(7x9)+(9x9)+(9x5)+(5x2)")
        weight_calc.scale(0.66)
        total_weights = Tex("199")
        total_weights.set_color(TEAL)
        weight_calc.next_to(weights, RIGHT)
        total_weights.next_to(weights, RIGHT)
        self.play(Write(weight_calc))
        self.play(ReplacementTransform(weight_calc, total_weights))

        # biases
        bias_calc = Tex("7+9+9+5+2")
        total_biases = Tex("32")
        total_biases.set_color(TEAL)
        bias_calc.next_to(biases, RIGHT)
        total_biases.next_to(biases, RIGHT)
        self.play(Write(bias_calc))
        self.play(ReplacementTransform(bias_calc, total_biases))

        # parameters
        param_calc = Tex("199+32")
        total_params = Tex("231")
        total_params.set_color(TEAL)
        param_calc.next_to(params, RIGHT)
        total_params.next_to(params, RIGHT)
        self.play(Write(param_calc))
        self.play(ReplacementTransform(param_calc, total_params))

        # network + stats go away
        network = VGroup(self.layers, self.edge_groups)
        calculations = VGroup(total_weights, total_biases, total_params)
        group = VGroup(network, calculations)
        self.play(FadeOut(group))

        # FOURTH ARCHITECTURE
        arch_4 = Tex("7, 10, 10, 10, 7")
        arch_4.set_color(TEAL)
        arch_4.shift(UP + 2 * RIGHT)
        arch_4.next_to(layer_sizes, RIGHT)
        self.play(ReplacementTransform(arch_3, arch_4))
        self.add_neurons([7, 10, 10, 10, 7])
        self.edge_security()
        self.continual_animation()

        # weights
        weight_calc = Tex("(7x10)+(10x10)+(10x10)+(10x7)")
        weight_calc.scale(0.5)
        total_weights = Tex("340")
        total_weights.set_color(TEAL)
        weight_calc.next_to(weights, RIGHT)
        total_weights.next_to(weights, RIGHT)
        self.play(Write(weight_calc))
        self.play(ReplacementTransform(weight_calc, total_weights))

        # biases
        bias_calc = Tex("7+10+10+10+7")
        total_biases = Tex("44")
        total_biases.set_color(TEAL)
        bias_calc.next_to(biases, RIGHT)
        total_biases.next_to(biases, RIGHT)
        self.play(Write(bias_calc))
        self.play(ReplacementTransform(bias_calc, total_biases))

        # parameters
        param_calc = Tex("340+44")
        total_params = Tex("384")
        total_params.set_color(TEAL)
        param_calc.next_to(params, RIGHT)
        total_params.next_to(params, RIGHT)
        self.play(Write(param_calc))
        self.play(ReplacementTransform(param_calc, total_params))

        # network + stats go away
        network = VGroup(self.layers, self.edge_groups)
        calculations = VGroup(total_weights, total_biases, total_params)
        group = VGroup(network, calculations)
        self.play(FadeOut(group))

        # FIFTH ARCHITECTURE
        arch_5 = Tex("9, 12, 12, 12, 9")
        arch_5.set_color(TEAL)
        arch_5.shift(UP + 2 * RIGHT)
        arch_5.next_to(layer_sizes, RIGHT)
        self.play(ReplacementTransform(arch_4, arch_5))
        self.add_neurons([9, 12, 12, 12, 9])
        self.edge_security()
        self.continual_animation()

        # weights
        weight_calc = Tex("(9x12)+(12x12)+(12x12)+(12x9)")
        weight_calc.scale(0.5)
        total_weights = Tex("504")
        total_weights.set_color(TEAL)
        weight_calc.next_to(weights, RIGHT)
        total_weights.next_to(weights, RIGHT)
        self.play(Write(weight_calc))
        self.play(ReplacementTransform(weight_calc, total_weights))

        # biases
        bias_calc = Tex("9+12+12+12+9")
        total_biases = Tex("54")
        total_biases.set_color(TEAL)
        bias_calc.next_to(biases, RIGHT)
        total_biases.next_to(biases, RIGHT)
        self.play(Write(bias_calc))
        self.play(ReplacementTransform(bias_calc, total_biases))

        # parameters
        param_calc = Tex("504+54")
        total_params = Tex("558")
        total_params.set_color(TEAL)
        param_calc.next_to(params, RIGHT)
        total_params.next_to(params, RIGHT)
        self.play(Write(param_calc))
        self.play(ReplacementTransform(param_calc, total_params))

        # network + stats go away
        network = VGroup(self.layers, self.edge_groups)
        calculations = VGroup(total_weights, total_biases, total_params)
        group = VGroup(network, calculations)
        self.play(FadeOut(group))





