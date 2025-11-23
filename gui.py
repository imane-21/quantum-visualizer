"""Tkinter GUI for the Quantum Calculator with Modern Style & State Builder."""
import tkinter as tk
from tkinter import messagebox
from typing import Optional
import numpy as np

# Imports pour le Style et Matplotlib
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

try:
    from . import quantum_utils as qu
except Exception:
    import quantum_utils as qu

# Configuration de Matplotlib
plt.style.use('dark_background')

class QuantumCalculator(ttk.Frame):
    def __init__(self, master: Optional[ttk.Window] = None):
        super().__init__(master, padding=20)
        self.pack(fill=BOTH, expand=True)
        
        # Variables
        self.num_qubits_var = ttk.IntVar(value=2)
        self.state_builder_vars = [] 
        self.steps = [] 
        
        # Initialisation graphs
        self.figure = None
        self.ax = None
        self.canvas_plot = None

        self.create_layout()
        
        # Initialisation état par défaut
        self.set_state_vector_text(np.array([1, 0, 0, 0], dtype=complex))
        self.update_output(np.array([1, 0, 0, 0], dtype=complex))

    def create_layout(self):
        left_panel = ttk.Frame(self)
        left_panel.pack(side=LEFT, fill=BOTH, expand=True, padx=(0, 10))
        
        right_panel = ttk.Frame(self)
        right_panel.pack(side=RIGHT, fill=BOTH, expand=True)

        self.create_input_section(left_panel)
        self.create_gate_section(left_panel)
        self.create_circuit_section(left_panel)
        self.create_viz_section(right_panel)

    def create_input_section(self, parent):
        # --- Section 1 : Initialisation ---
        frame = ttk.Labelframe(parent, text="1. State Initialization", padding=10, bootstyle="info")
        frame.pack(fill=X, pady=(0, 10))

        # Zone de saisie manuelle
        lbl = ttk.Label(frame, text="Manual Vector Input:", font=("Helvetica", 8, "italic"))
        lbl.pack(anchor=W)
        
        input_row = ttk.Frame(frame)
        input_row.pack(fill=X, pady=(0, 10))
        
        self.state_entry = ttk.Entry(input_row, bootstyle="dark")
        self.state_entry.pack(side=LEFT, fill=X, expand=True)
        ttk.Button(input_row, text="Update Plot", command=self.manual_update, bootstyle="info").pack(side=LEFT, padx=5)

        # --- LE CLAVIER / BUILDER ---
        tabs = ttk.Notebook(frame, bootstyle="secondary")
        tabs.pack(fill=X)

        # Tab 1: Builder
        tab_builder = ttk.Frame(tabs, padding=10)
        tabs.add(tab_builder, text="Builder (Tensor Product)")
        
        row1 = ttk.Frame(tab_builder)
        row1.pack(fill=X)
        ttk.Label(row1, text="Qubits:").pack(side=LEFT)
        self.spin_qubits = ttk.Spinbox(row1, from_=1, to=4, textvariable=self.num_qubits_var, width=5, command=self.update_builder_rows)
        self.spin_qubits.pack(side=LEFT, padx=5)
        ttk.Button(row1, text="Set Slots", command=self.update_builder_rows, bootstyle="outline-secondary", width=8).pack(side=LEFT)

        self.builder_frame = ttk.Frame(tab_builder)
        self.builder_frame.pack(fill=X, pady=5)
        self.update_builder_rows() 

        ttk.Button(tab_builder, text="Apply & Calculate", command=self.apply_tensor_build, bootstyle="success").pack(fill=X, pady=5)

        # Tab 2: Special States
        tab_special = ttk.Frame(tabs, padding=10)
        tabs.add(tab_special, text="Bell / Entangled")
        
        grid_frm = ttk.Frame(tab_special)
        grid_frm.pack(fill=X)
        
        # Les 4 États de Bell
        ttk.Button(grid_frm, text="Φ+ (|00>+|11>)", command=lambda: self.set_bell_state("phi+"), bootstyle="warning-outline").grid(row=0, column=0, padx=2, pady=2, sticky=EW)
        ttk.Button(grid_frm, text="Φ- (|00>-|11>)", command=lambda: self.set_bell_state("phi-"), bootstyle="warning-outline").grid(row=0, column=1, padx=2, pady=2, sticky=EW)
        ttk.Button(grid_frm, text="Ψ+ (|01>+|10>)", command=lambda: self.set_bell_state("psi+"), bootstyle="warning-outline").grid(row=1, column=0, padx=2, pady=2, sticky=EW)
        ttk.Button(grid_frm, text="Ψ- (|01>-|10>)", command=lambda: self.set_bell_state("psi-"), bootstyle="warning-outline").grid(row=1, column=1, padx=2, pady=2, sticky=EW)
        
        # GHZ
        ttk.Button(grid_frm, text="GHZ (3-qubit)", command=lambda: self.set_ghz_state(), bootstyle="danger-outline").grid(row=2, column=0, columnspan=2, padx=2, pady=5, sticky=EW)
        
        grid_frm.columnconfigure(0, weight=1)
        grid_frm.columnconfigure(1, weight=1)
        
        # Outils
        tool_row = ttk.Frame(frame)
        tool_row.pack(fill=X, pady=(10, 0))
        ttk.Button(tool_row, text="Check Norm", command=self.check_norm, bootstyle="secondary-outline", width=12).pack(side=LEFT, padx=2)
        ttk.Button(tool_row, text="Normalize", command=self.normalize_state, bootstyle="secondary-outline", width=12).pack(side=LEFT, padx=2)

    def update_builder_rows(self):
        for widget in self.builder_frame.winfo_children():
            widget.destroy()
        self.state_builder_vars = []
        
        try:
            n = int(self.num_qubits_var.get())
        except:
            n = 1
            
        for i in range(n):
            row = ttk.Frame(self.builder_frame)
            row.pack(fill=X, pady=2)
            ttk.Label(row, text=f"q{i}:", width=4).pack(side=LEFT)
            var = ttk.StringVar(value="|0>")
            self.state_builder_vars.append(var)
            combo = ttk.Combobox(row, textvariable=var, values=["|0>", "|1>", "|+>", "|->", "|i>", "|-i>"], state="readonly", width=8)
            combo.pack(side=LEFT, fill=X, expand=True)

    def create_gate_section(self, parent):
        frame = ttk.Labelframe(parent, text="2. Single Operations", padding=10, bootstyle="primary")
        frame.pack(fill=X, pady=(0, 10))

        row = ttk.Frame(frame)
        row.pack(fill=X)

        ttk.Label(row, text="Gate:").pack(side=LEFT)
        self.gate_var = ttk.StringVar(value='H')
        gates = ['X', 'Y', 'Z', 'H', 'S', 'T']
        ttk.Combobox(row, values=gates, textvariable=self.gate_var, state='readonly', width=5).pack(side=LEFT, padx=5)

        ttk.Label(row, text="Target q:").pack(side=LEFT)
        self.target_var = ttk.StringVar(value='0')
        ttk.Entry(row, textvariable=self.target_var, width=4).pack(side=LEFT, padx=5)

        ttk.Button(row, text="Apply", command=self.apply_gate, bootstyle="primary").pack(side=LEFT, padx=10)

    def create_circuit_section(self, parent):
        frame = ttk.Labelframe(parent, text="3. Circuit Builder", padding=10, bootstyle="success")
        frame.pack(fill=BOTH, expand=True)

        # Controls
        row = ttk.Frame(frame)
        row.pack(fill=X)
        
        self.cb_gate_var = ttk.StringVar(value='H')
        ttk.Combobox(row, values=['X', 'H', 'Z', 'S', 'T', 'CNOT'], textvariable=self.cb_gate_var, state='readonly', width=6).pack(side=LEFT)
        ttk.Label(row, text="Tgt:").pack(side=LEFT, padx=(5,0))
        self.cb_target = ttk.StringVar(value='0')
        ttk.Entry(row, textvariable=self.cb_target, width=3).pack(side=LEFT)
        ttk.Label(row, text="Ctl:").pack(side=LEFT, padx=(5,0))
        self.cb_control = ttk.StringVar(value='')
        ttk.Entry(row, textvariable=self.cb_control, width=3).pack(side=LEFT)
        
        # BOUTONS D'ACTION (Ajout de DEL)
        ttk.Button(row, text="+", command=self.cb_add_step, width=3, bootstyle="success-outline").pack(side=LEFT, padx=5)
        ttk.Button(row, text="Del", command=self.cb_delete_step, width=3, bootstyle="warning-outline").pack(side=LEFT, padx=2)
        ttk.Button(row, text="Clr", command=self.cb_clear_steps, width=3, bootstyle="danger-outline").pack(side=LEFT, padx=2)

        # Listbox
        self.steps_listbox = tk.Listbox(frame, height=6, bg="#2b2b2b", fg="#dddddd", borderwidth=0, highlightthickness=0)
        self.steps_listbox.pack(fill=BOTH, expand=True, pady=5)
        
        ttk.Button(frame, text="Run Circuit", command=self.cb_apply_circuit, bootstyle="success").pack(fill=X)

    def create_viz_section(self, parent):
        # Graphique
        self.figure, self.ax = plt.subplots(figsize=(5, 4), dpi=100)
        self.figure.patch.set_facecolor('#222222') 
        self.ax.set_facecolor('#222222')
        self.ax.tick_params(colors='white')
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')
        self.ax.title.set_color('white')
        for spine in self.ax.spines.values():
            spine.set_edgecolor('#555555')

        self.canvas_plot = FigureCanvasTkAgg(self.figure, master=parent)
        self.canvas_plot.get_tk_widget().pack(fill=BOTH, expand=True, pady=(0, 10))

        # Logs
        log_frame = ttk.Labelframe(parent, text="Amplitude Logs", padding=5, bootstyle="secondary")
        log_frame.pack(fill=X, side=BOTTOM)
        
        self.output_text = tk.Text(log_frame, height=5, bg="#000000", fg="#00ff00", borderwidth=0, font=("Consolas", 9))
        self.output_text.pack(fill=X)

    # --- LOGIC ---

    def manual_update(self):
        s = self.state_entry.get()
        try:
            st = qu.parse_state_vector(s)
            self.update_output(st)
        except Exception as e:
            messagebox.showerror("Error", f"Invalid state: {e}")

    def apply_tensor_build(self):
        parts = []
        try:
            for i, var in enumerate(self.state_builder_vars):
                parts.append(var.get())
            
            current_state = self._parse_single_ket(parts[0])
            for p in parts[1:]:
                next_st = self._parse_single_ket(p)
                current_state = np.kron(current_state, next_st)
            
            self.set_state_vector_text(current_state)
            self.update_output(current_state)
        except Exception as e:
            messagebox.showerror("Builder Error", str(e))

    def _parse_single_ket(self, k_str):
        if k_str == "|0>": return np.array([1, 0], dtype=complex)
        if k_str == "|1>": return np.array([0, 1], dtype=complex)
        if k_str == "|+>": return np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
        if k_str == "|->": return np.array([1/np.sqrt(2), -1/np.sqrt(2)], dtype=complex)
        if k_str == "|i>": return np.array([1/np.sqrt(2), 1j/np.sqrt(2)], dtype=complex)
        if k_str == "|-i>": return np.array([1/np.sqrt(2), -1j/np.sqrt(2)], dtype=complex)
        return np.array([1, 0], dtype=complex)

    def set_bell_state(self, type_):
        self.num_qubits_var.set(2)
        self.update_builder_rows() 
        sq2 = 1/np.sqrt(2)
        if type_ == "phi+": st = np.array([sq2, 0, 0, sq2], dtype=complex)
        elif type_ == "phi-": st = np.array([sq2, 0, 0, -sq2], dtype=complex)
        elif type_ == "psi+": st = np.array([0, sq2, sq2, 0], dtype=complex)
        elif type_ == "psi-": st = np.array([0, sq2, -sq2, 0], dtype=complex)
        self.set_state_vector_text(st)
        self.update_output(st)

    def set_ghz_state(self):
        self.num_qubits_var.set(3)
        self.update_builder_rows()
        sq2 = 1/np.sqrt(2)
        st = np.zeros(8, dtype=complex)
        st[0] = sq2
        st[7] = sq2
        self.set_state_vector_text(st)
        self.update_output(st)

    def set_state_vector_text(self, state):
        self.state_entry.delete(0, tk.END)
        s_str = ",".join([f"{x.real:.3f}+{x.imag:.3f}j" if x.imag != 0 else f"{x.real:.3f}" for x in state])
        self.state_entry.insert(0, s_str)

    def parse_state(self):
        s = self.state_entry.get()
        try:
            state = qu.parse_state_vector(s)
        except Exception as e:
            messagebox.showerror("Parse error", str(e))
            return None
        return state

    def update_output(self, state: np.ndarray):
        self.output_text.delete('1.0', tk.END)
        num_q = int(np.log2(len(state)))
        for i, amp in enumerate(state):
            if abs(amp) > 0.001:
                bin_idx = format(i, f'0{num_q}b')
                self.output_text.insert(tk.END, f"|{bin_idx}> : {amp:.3f}\n")
        
        if self.ax is None: return
        self.ax.clear()
        probs = qu.computational_probabilities(state)
        num_states = len(state)
        if num_states > 32: 
            self.ax.text(0.5, 0.5, "Too many states to plot", ha='center', va='center', color='white')
            self.canvas_plot.draw()
            return

        labels = [f"|{format(i, f'0{num_q}b')}>" for i in range(num_states)]
        bars = self.ax.bar(labels, probs, color='#00d2ff', alpha=0.8, edgecolor='white', linewidth=0.5)
        self.ax.set_ylim(0, 1.1)
        self.ax.set_title(f"Quantum State Probabilities ({num_q} Qubits)")
        if num_states > 8: self.ax.tick_params(axis='x', rotation=45, labelsize=8)
        for bar in bars:
            height = bar.get_height()
            if height > 0.01:
                self.ax.text(bar.get_x() + bar.get_width()/2., height + 0.02, f'{height:.2f}', ha='center', va='bottom', color='#00d2ff', fontsize=8, fontweight='bold')
        self.canvas_plot.draw()

    # --- WRAPPERS ---
    def check_norm(self):
        st = self.parse_state()
        if st is not None: messagebox.showinfo("Norm", f"{qu.norm(st):.4f}")
    
    def normalize_state(self):
        st = self.parse_state()
        if st is not None:
            nst = qu.normalize(st)
            self.set_state_vector_text(nst)
            self.update_output(nst)

    def apply_gate(self):
        st = self.parse_state()
        if st is None: return
        try:
            num_q = int(np.log2(len(st)))
            tgt = int(self.target_var.get())
            g = qu.single_qubit_gate_by_name(self.gate_var.get())
            nst = qu.apply_gate(st, g, [tgt], num_q)
            self.set_state_vector_text(nst)
            self.update_output(nst)
        except Exception as e:
            messagebox.showerror("Gate Error", str(e))

    def cb_add_step(self):
        g = self.cb_gate_var.get()
        t = self.cb_target.get()
        c = self.cb_control.get()
        self.steps.append({'gate': g, 'target': t, 'control': c})
        if g == 'CNOT':
            display_text = f"CNOT: Control q{c} -> Target q{t}"
        else:
            display_text = f"{g} gate on q{t}"
        self.steps_listbox.insert(tk.END, display_text)

    # NOUVELLE FONCTION DE SUPPRESSION
    def cb_delete_step(self):
        selection = self.steps_listbox.curselection()
        if not selection:
            return
        index = selection[0]
        self.steps.pop(index)
        self.steps_listbox.delete(index)

    def cb_clear_steps(self):
        self.steps = []
        self.steps_listbox.delete(0, tk.END)

    def cb_apply_circuit(self):
        st = self.parse_state()
        if st is None: return
        num_q = int(np.log2(len(st)))
        curr = st.copy()
        try:
            for step in self.steps:
                if step['gate'] == 'CNOT':
                    curr = qu.apply_cnot(curr, int(step['control']), int(step['target']), num_q)
                else:
                    g = qu.single_qubit_gate_by_name(step['gate'])
                    curr = qu.apply_gate(curr, g, [int(step['target'])], num_q)
            self.set_state_vector_text(curr)
            self.update_output(curr)
        except Exception as e:
            messagebox.showerror("Circuit Execution Error", str(e))

def run_app():
    app = ttk.Window(title="Quantum Visualizer Pro", themename="darkly")
    QuantumCalculator(app)
    app.mainloop()