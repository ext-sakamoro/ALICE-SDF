// =============================================================================
// SDF World Custom Editor
// =============================================================================
// Advanced inspector with Scene View Gizmos, performance monitoring,
// one-click presets, and interactive handles.
//
// Author: Moroya Sakamoto
// "5MB to render infinity. That's not a bug, that's mathematics."
// =============================================================================

using UnityEngine;
using UnityEditor;
using SdfUniverse;
using System.Collections.Generic;

namespace SdfUniverseEditor
{
    [CustomEditor(typeof(SdfWorld))]
    public class SdfWorldEditor : Editor
    {
        // =====================================================================
        // Constants
        // =====================================================================

        private const int INSTRUCTION_WARNING_THRESHOLD = 500;
        private const int INSTRUCTION_CRITICAL_THRESHOLD = 1000;
        private const float COMPILE_TIME_WARNING_MS = 5f;
        private const float COMPILE_TIME_CRITICAL_MS = 10f;

        // =====================================================================
        // State
        // =====================================================================

        private SdfWorld _world;
        private bool _showPerformance = true;
        private bool _showQuickActions = true;
        private bool _showPresets = true;
        private bool _showAdvanced = false;
        private bool _showDebug = false;
        private int _selectedObjectIndex = -1;

        // Styles
        private GUIStyle _headerStyle;
        private GUIStyle _subHeaderStyle;
        private GUIStyle _statusGoodStyle;
        private GUIStyle _statusWarningStyle;
        private GUIStyle _statusCriticalStyle;
        private GUIStyle _boxStyle;
        private bool _stylesInitialized = false;

        // Serialized Properties
        private SerializedProperty _skyEnabled;
        private SerializedProperty _skyRadius;
        private SerializedProperty _groundEnabled;
        private SerializedProperty _groundHeight;
        private SerializedProperty _objects;
        private SerializedProperty _globalSmoothness;
        private SerializedProperty _infiniteRepeat;
        private SerializedProperty _repeatSpacing;

        // =====================================================================
        // Icons (Unicode)
        // =====================================================================

        private const string ICON_SPHERE = "\u25CF";      // ●
        private const string ICON_BOX = "\u25A0";         // ■
        private const string ICON_TORUS = "\u25CB";       // ○
        private const string ICON_METABALL = "\u2B24";    // ⬤
        private const string ICON_PLANET = "\u2295";      // ⊕
        private const string ICON_COSMIC = "\u2726";      // ✦
        private const string ICON_CHECK = "\u2713";       // ✓
        private const string ICON_WARNING = "\u26A0";     // ⚠
        private const string ICON_CRITICAL = "\u2717";    // ✗
        private const string ICON_COMPILE = "\u27F3";     // ⟳
        private const string ICON_INSTRUCTION = "\u2192"; // →

        // =====================================================================
        // Unity Callbacks
        // =====================================================================

        void OnEnable()
        {
            _world = (SdfWorld)target;

            _skyEnabled = serializedObject.FindProperty("skyEnabled");
            _skyRadius = serializedObject.FindProperty("skyRadius");
            _groundEnabled = serializedObject.FindProperty("groundEnabled");
            _groundHeight = serializedObject.FindProperty("groundHeight");
            _objects = serializedObject.FindProperty("objects");
            _globalSmoothness = serializedObject.FindProperty("globalSmoothness");
            _infiniteRepeat = serializedObject.FindProperty("infiniteRepeat");
            _repeatSpacing = serializedObject.FindProperty("repeatSpacing");

            SceneView.duringSceneGui += OnSceneGUI;
        }

        void OnDisable()
        {
            SceneView.duringSceneGui -= OnSceneGUI;
        }

        // =====================================================================
        // Style Initialization
        // =====================================================================

        private void InitializeStyles()
        {
            if (_stylesInitialized) return;

            _headerStyle = new GUIStyle(EditorStyles.boldLabel)
            {
                fontSize = 14,
                alignment = TextAnchor.MiddleCenter,
                normal = { textColor = new Color(0.9f, 0.7f, 0.2f) }
            };

            _subHeaderStyle = new GUIStyle(EditorStyles.boldLabel)
            {
                fontSize = 12,
                normal = { textColor = new Color(0.7f, 0.9f, 1f) }
            };

            _statusGoodStyle = new GUIStyle(EditorStyles.label)
            {
                normal = { textColor = new Color(0.4f, 1f, 0.4f) },
                fontStyle = FontStyle.Bold
            };

            _statusWarningStyle = new GUIStyle(EditorStyles.label)
            {
                normal = { textColor = new Color(1f, 0.8f, 0.2f) },
                fontStyle = FontStyle.Bold
            };

            _statusCriticalStyle = new GUIStyle(EditorStyles.label)
            {
                normal = { textColor = new Color(1f, 0.4f, 0.4f) },
                fontStyle = FontStyle.Bold
            };

            _boxStyle = new GUIStyle("box")
            {
                padding = new RectOffset(10, 10, 10, 10)
            };

            _stylesInitialized = true;
        }

        // =====================================================================
        // Main Inspector GUI
        // =====================================================================

        public override void OnInspectorGUI()
        {
            InitializeStyles();
            serializedObject.Update();

            DrawHeader();
            DrawPerformanceSection();
            DrawQuickActionsSection();
            DrawPresetsSection();
            DrawWorldSettingsSection();
            DrawObjectsSection();
            DrawAdvancedSection();
            DrawDebugSection();
            DrawFooter();

            serializedObject.ApplyModifiedProperties();

            if (Application.isPlaying)
            {
                Repaint();
            }
        }

        // =====================================================================
        // Header
        // =====================================================================

        private new void DrawHeader()
        {
            EditorGUILayout.Space(5);

            // Title Box
            EditorGUILayout.BeginVertical(_boxStyle);
            EditorGUILayout.LabelField($"{ICON_COSMIC} SDF World {ICON_COSMIC}", _headerStyle);
            EditorGUILayout.LabelField("\"5MB to render infinity\"", EditorStyles.centeredGreyMiniLabel);
            EditorGUILayout.EndVertical();

            EditorGUILayout.Space(5);
        }

        // =====================================================================
        // Performance Section
        // =====================================================================

        private void DrawPerformanceSection()
        {
            _showPerformance = EditorGUILayout.Foldout(_showPerformance,
                $"{ICON_COMPILE} Performance Monitor", true, EditorStyles.foldoutHeader);

            if (!_showPerformance) return;

            EditorGUILayout.BeginVertical(_boxStyle);

            if (Application.isPlaying)
            {
                // JIT Status
                DrawStatusRow("JIT Status",
                    _world.IsReady ? $"{ICON_CHECK} Compiled" : $"{ICON_COMPILE} Building...",
                    _world.IsReady ? _statusGoodStyle : _statusWarningStyle);

                // Compile Time
                float compileTime = _world.LastCompileTimeMs;
                GUIStyle compileStyle = compileTime < COMPILE_TIME_WARNING_MS ? _statusGoodStyle :
                                        compileTime < COMPILE_TIME_CRITICAL_MS ? _statusWarningStyle :
                                        _statusCriticalStyle;
                string compileIcon = compileTime < COMPILE_TIME_WARNING_MS ? ICON_CHECK :
                                    compileTime < COMPILE_TIME_CRITICAL_MS ? ICON_WARNING :
                                    ICON_CRITICAL;
                DrawStatusRow("Compile Time", $"{compileIcon} {compileTime:F2} ms", compileStyle);

                // Instruction Count
                if (_world.CompiledWorld != null && _world.CompiledWorld.IsValid)
                {
                    int instructions = (int)_world.CompiledWorld.InstructionCount;
                    GUIStyle instrStyle = instructions < INSTRUCTION_WARNING_THRESHOLD ? _statusGoodStyle :
                                          instructions < INSTRUCTION_CRITICAL_THRESHOLD ? _statusWarningStyle :
                                          _statusCriticalStyle;
                    string instrIcon = instructions < INSTRUCTION_WARNING_THRESHOLD ? ICON_CHECK :
                                      instructions < INSTRUCTION_CRITICAL_THRESHOLD ? ICON_WARNING :
                                      ICON_CRITICAL;
                    DrawStatusRow("Instructions", $"{instrIcon} {instructions} ops", instrStyle);

                    // Performance Warnings
                    if (instructions >= INSTRUCTION_CRITICAL_THRESHOLD)
                    {
                        EditorGUILayout.Space(5);
                        EditorGUILayout.HelpBox(
                            $"{ICON_CRITICAL} High instruction count may impact performance.\n" +
                            "Consider reducing object count or simplifying shapes.",
                            MessageType.Warning);
                    }
                    else if (instructions >= INSTRUCTION_WARNING_THRESHOLD)
                    {
                        EditorGUILayout.Space(5);
                        EditorGUILayout.HelpBox(
                            $"{ICON_WARNING} Moderate instruction count.\n" +
                            "Performance is acceptable but monitor frame rate.",
                            MessageType.Info);
                    }
                }

                // Object Count
                DrawStatusRow("Objects", $"{ICON_INSTRUCTION} {_world.objects.Count} shapes", EditorStyles.label);
            }
            else
            {
                EditorGUILayout.HelpBox("Enter Play Mode to see real-time performance stats.", MessageType.Info);
            }

            EditorGUILayout.EndVertical();
            EditorGUILayout.Space(5);
        }

        private void DrawStatusRow(string label, string value, GUIStyle valueStyle)
        {
            EditorGUILayout.BeginHorizontal();
            EditorGUILayout.LabelField(label, GUILayout.Width(100));
            EditorGUILayout.LabelField(value, valueStyle);
            EditorGUILayout.EndHorizontal();
        }

        // =====================================================================
        // Quick Actions Section
        // =====================================================================

        private void DrawQuickActionsSection()
        {
            _showQuickActions = EditorGUILayout.Foldout(_showQuickActions,
                $"{ICON_SPHERE} Quick Add Shapes", true, EditorStyles.foldoutHeader);

            if (!_showQuickActions) return;

            EditorGUILayout.BeginVertical(_boxStyle);

            // Row 1: Basic Shapes
            EditorGUILayout.BeginHorizontal();
            if (GUILayout.Button($"{ICON_SPHERE} Sphere", GUILayout.Height(30)))
            {
                AddObject(SdfShapeType.Sphere, "Sphere");
            }
            if (GUILayout.Button($"{ICON_BOX} Box", GUILayout.Height(30)))
            {
                AddObject(SdfShapeType.Box, "Box");
            }
            if (GUILayout.Button($"{ICON_TORUS} Torus", GUILayout.Height(30)))
            {
                AddObject(SdfShapeType.Torus, "Torus");
            }
            EditorGUILayout.EndHorizontal();

            // Row 2: Advanced Shapes
            EditorGUILayout.BeginHorizontal();
            if (GUILayout.Button($"{ICON_METABALL} Metaball", GUILayout.Height(30)))
            {
                AddObject(SdfShapeType.Metaball, "Metaball");
            }
            if (GUILayout.Button($"{ICON_TORUS} Cylinder", GUILayout.Height(30)))
            {
                AddObject(SdfShapeType.Cylinder, "Cylinder");
            }
            if (GUILayout.Button($"{ICON_BOX} Capsule", GUILayout.Height(30)))
            {
                AddObject(SdfShapeType.Capsule, "Capsule");
            }
            EditorGUILayout.EndHorizontal();

            EditorGUILayout.Space(5);

            // Control Buttons
            EditorGUILayout.BeginHorizontal();
            GUI.backgroundColor = new Color(0.3f, 0.7f, 1f);
            if (GUILayout.Button($"{ICON_COMPILE} Rebuild World", GUILayout.Height(25)))
            {
                _world.RebuildWorld();
            }
            GUI.backgroundColor = new Color(1f, 0.5f, 0.5f);
            if (GUILayout.Button($"{ICON_CRITICAL} Clear All", GUILayout.Height(25)))
            {
                if (EditorUtility.DisplayDialog("Clear Objects",
                    "Remove all SDF objects from the world?", "Yes", "Cancel"))
                {
                    Undo.RecordObject(_world, "Clear All Objects");
                    _world.objects.Clear();
                    _selectedObjectIndex = -1;
                    EditorUtility.SetDirty(_world);
                    serializedObject.Update();
                }
            }
            GUI.backgroundColor = Color.white;
            EditorGUILayout.EndHorizontal();

            EditorGUILayout.EndVertical();
            EditorGUILayout.Space(5);
        }

        // =====================================================================
        // Presets Section
        // =====================================================================

        private void DrawPresetsSection()
        {
            _showPresets = EditorGUILayout.Foldout(_showPresets,
                $"{ICON_PLANET} Scene Presets", true, EditorStyles.foldoutHeader);

            if (!_showPresets) return;

            EditorGUILayout.BeginVertical(_boxStyle);
            EditorGUILayout.LabelField("One-Click Scene Setup", EditorStyles.centeredGreyMiniLabel);
            EditorGUILayout.Space(5);

            EditorGUILayout.BeginHorizontal();
            GUI.backgroundColor = new Color(1f, 0.6f, 0.2f);
            if (GUILayout.Button($"{ICON_PLANET} Planet", GUILayout.Height(35)))
            {
                CreatePlanetPreset();
            }
            GUI.backgroundColor = new Color(0.8f, 0.5f, 1f);
            if (GUILayout.Button($"{ICON_COSMIC} Cosmic", GUILayout.Height(35)))
            {
                CreateCosmicPreset();
            }
            GUI.backgroundColor = new Color(0.4f, 0.8f, 0.4f);
            if (GUILayout.Button($"{ICON_TORUS} Abstract", GUILayout.Height(35)))
            {
                CreateAbstractPreset();
            }
            GUI.backgroundColor = Color.white;
            EditorGUILayout.EndHorizontal();

            EditorGUILayout.BeginHorizontal();
            GUI.backgroundColor = new Color(0.4f, 0.7f, 1f);
            if (GUILayout.Button($"{ICON_BOX} Terrain", GUILayout.Height(35)))
            {
                CreateTerrainPreset();
            }
            GUI.backgroundColor = new Color(1f, 0.8f, 0.3f);
            if (GUILayout.Button($"{ICON_METABALL} Metaballs", GUILayout.Height(35)))
            {
                CreateMetaballPreset();
            }
            GUI.backgroundColor = new Color(0.9f, 0.3f, 0.5f);
            if (GUILayout.Button($"{ICON_SPHERE} Atom", GUILayout.Height(35)))
            {
                CreateAtomPreset();
            }
            GUI.backgroundColor = Color.white;
            EditorGUILayout.EndHorizontal();

            EditorGUILayout.EndVertical();
            EditorGUILayout.Space(5);
        }

        // =====================================================================
        // World Settings Section
        // =====================================================================

        private void DrawWorldSettingsSection()
        {
            EditorGUILayout.LabelField("World Settings", _subHeaderStyle);
            EditorGUILayout.BeginVertical(_boxStyle);

            // Sky
            EditorGUILayout.BeginHorizontal();
            EditorGUILayout.PropertyField(_skyEnabled, new GUIContent("Sky Sphere"));
            if (_skyEnabled.boolValue)
            {
                EditorGUILayout.PropertyField(_skyRadius, GUIContent.none, GUILayout.Width(60));
            }
            EditorGUILayout.EndHorizontal();

            // Ground
            EditorGUILayout.BeginHorizontal();
            EditorGUILayout.PropertyField(_groundEnabled, new GUIContent("Ground Plane"));
            if (_groundEnabled.boolValue)
            {
                EditorGUILayout.PropertyField(_groundHeight, GUIContent.none, GUILayout.Width(60));
            }
            EditorGUILayout.EndHorizontal();

            EditorGUILayout.Space(5);

            // Global Smoothness
            EditorGUILayout.PropertyField(_globalSmoothness, new GUIContent("Global Smoothness"));

            // Infinite Repeat
            EditorGUILayout.PropertyField(_infiniteRepeat, new GUIContent("Infinite Repeat"));
            if (_infiniteRepeat.boolValue)
            {
                EditorGUI.indentLevel++;
                EditorGUILayout.PropertyField(_repeatSpacing, new GUIContent("Repeat Spacing"));
                EditorGUILayout.HelpBox(
                    $"{ICON_COSMIC} Infinite repeat enabled!\n" +
                    "Objects tile infinitely in all directions.",
                    MessageType.Info);
                EditorGUI.indentLevel--;
            }

            EditorGUILayout.EndVertical();
            EditorGUILayout.Space(5);
        }

        // =====================================================================
        // Objects Section
        // =====================================================================

        private void DrawObjectsSection()
        {
            EditorGUILayout.BeginHorizontal();
            EditorGUILayout.LabelField($"Objects ({_world.objects.Count})", _subHeaderStyle);
            if (GUILayout.Button("Collapse All", GUILayout.Width(80)))
            {
                _selectedObjectIndex = -1;
            }
            EditorGUILayout.EndHorizontal();

            EditorGUILayout.BeginVertical(_boxStyle);

            if (_world.objects.Count == 0)
            {
                EditorGUILayout.HelpBox("No objects in world. Use Quick Add or Presets above.", MessageType.Info);
            }
            else
            {
                for (int i = 0; i < _world.objects.Count; i++)
                {
                    DrawObjectEntry(i);
                }
            }

            EditorGUILayout.EndVertical();
            EditorGUILayout.Space(5);
        }

        private void DrawObjectEntry(int index)
        {
            // Bounds check
            if (index < 0 || index >= _world.objects.Count) return;

            var obj = _world.objects[index];
            if (obj == null) return;

            string icon = GetShapeIcon(obj.shapeType);

            EditorGUILayout.BeginHorizontal();

            // Foldout
            bool isSelected = _selectedObjectIndex == index;
            if (GUILayout.Button(isSelected ? "v" : ">", GUILayout.Width(20)))
            {
                _selectedObjectIndex = isSelected ? -1 : index;
            }

            // Icon and Name
            EditorGUILayout.LabelField($"{icon} {obj.name}", GUILayout.Width(150));

            // Position summary
            EditorGUILayout.LabelField($"({obj.position.x:F1}, {obj.position.y:F1}, {obj.position.z:F1})",
                EditorStyles.miniLabel);

            // Delete button
            GUI.backgroundColor = new Color(1f, 0.5f, 0.5f);
            if (GUILayout.Button("X", GUILayout.Width(20)))
            {
                Undo.RecordObject(_world, "Delete SDF Object");
                _world.objects.RemoveAt(index);
                if (_selectedObjectIndex >= _world.objects.Count)
                {
                    _selectedObjectIndex = _world.objects.Count - 1;
                }
                EditorUtility.SetDirty(_world);
                serializedObject.Update();
                EditorGUILayout.EndHorizontal();
                return;
            }
            GUI.backgroundColor = Color.white;

            EditorGUILayout.EndHorizontal();

            // Expanded view
            if (_selectedObjectIndex == index)
            {
                EditorGUI.indentLevel += 2;

                // Ensure serialized property is in sync and in bounds
                if (_objects != null && index < _objects.arraySize)
                {
                    var objProp = _objects.GetArrayElementAtIndex(index);
                    if (objProp != null)
                    {
                        EditorGUILayout.PropertyField(objProp.FindPropertyRelative("name"));
                        EditorGUILayout.PropertyField(objProp.FindPropertyRelative("shapeType"));
                        EditorGUILayout.PropertyField(objProp.FindPropertyRelative("position"));
                        EditorGUILayout.PropertyField(objProp.FindPropertyRelative("radius"));

                        if (obj.shapeType == SdfShapeType.Torus)
                        {
                            EditorGUILayout.PropertyField(objProp.FindPropertyRelative("majorRadius"));
                            EditorGUILayout.PropertyField(objProp.FindPropertyRelative("minorRadius"));
                        }
                        else if (obj.shapeType == SdfShapeType.Box)
                        {
                            EditorGUILayout.PropertyField(objProp.FindPropertyRelative("halfExtents"));
                        }
                    }
                }

                EditorGUI.indentLevel -= 2;
                EditorGUILayout.Space(3);
            }
        }

        private string GetShapeIcon(SdfShapeType shapeType)
        {
            return shapeType switch
            {
                SdfShapeType.Sphere => ICON_SPHERE,
                SdfShapeType.Box => ICON_BOX,
                SdfShapeType.Torus => ICON_TORUS,
                SdfShapeType.Metaball => ICON_METABALL,
                SdfShapeType.Cylinder => ICON_TORUS,
                SdfShapeType.Capsule => ICON_BOX,
                _ => ICON_SPHERE
            };
        }

        // =====================================================================
        // Advanced Section
        // =====================================================================

        private void DrawAdvancedSection()
        {
            _showAdvanced = EditorGUILayout.Foldout(_showAdvanced,
                $"{ICON_INSTRUCTION} Advanced Options", true, EditorStyles.foldoutHeader);

            if (!_showAdvanced) return;

            EditorGUILayout.BeginVertical(_boxStyle);

            EditorGUILayout.LabelField("Performance Thresholds", EditorStyles.boldLabel);
            EditorGUILayout.HelpBox(
                $"Instruction Warning: {INSTRUCTION_WARNING_THRESHOLD} ops\n" +
                $"Instruction Critical: {INSTRUCTION_CRITICAL_THRESHOLD} ops\n" +
                $"Compile Warning: {COMPILE_TIME_WARNING_MS} ms\n" +
                $"Compile Critical: {COMPILE_TIME_CRITICAL_MS} ms",
                MessageType.None);

            EditorGUILayout.Space(5);

            EditorGUILayout.LabelField("Scene View Gizmos", EditorStyles.boldLabel);
            EditorGUILayout.LabelField("Select objects in the Objects list to show handles.", EditorStyles.miniLabel);
            EditorGUILayout.LabelField("Drag handles in Scene View to move objects.", EditorStyles.miniLabel);

            EditorGUILayout.EndVertical();
            EditorGUILayout.Space(5);
        }

        // =====================================================================
        // Debug Section
        // =====================================================================

        private void DrawDebugSection()
        {
            _showDebug = EditorGUILayout.Foldout(_showDebug,
                $"{ICON_WARNING} Debug Info", true, EditorStyles.foldoutHeader);

            if (!_showDebug) return;

            EditorGUILayout.BeginVertical(_boxStyle);

            EditorGUILayout.LabelField("Component Status", EditorStyles.boldLabel);
            EditorGUILayout.LabelField($"SdfWorld: {(_world != null ? "Valid" : "NULL")}");
            EditorGUILayout.LabelField($"CompiledWorld: {(_world.CompiledWorld != null ? "Valid" : "NULL")}");
            EditorGUILayout.LabelField($"IsReady: {_world.IsReady}");
            EditorGUILayout.LabelField($"Object Count: {_world.objects.Count}");

            if (Application.isPlaying && _world.CompiledWorld != null)
            {
                EditorGUILayout.Space(5);
                EditorGUILayout.LabelField("Compiled World", EditorStyles.boldLabel);
                EditorGUILayout.LabelField($"IsValid: {_world.CompiledWorld.IsValid}");
                EditorGUILayout.LabelField($"Instructions: {_world.CompiledWorld.InstructionCount}");
            }

            EditorGUILayout.EndVertical();
        }

        // =====================================================================
        // Footer
        // =====================================================================

        private void DrawFooter()
        {
            EditorGUILayout.Space(10);
            EditorGUILayout.BeginHorizontal();
            GUILayout.FlexibleSpace();
            EditorGUILayout.LabelField("ALICE-SDF v1.0", EditorStyles.centeredGreyMiniLabel);
            GUILayout.FlexibleSpace();
            EditorGUILayout.EndHorizontal();
        }

        // =====================================================================
        // Scene View Gizmos
        // =====================================================================

        private void OnSceneGUI(SceneView sceneView)
        {
            if (_world == null || _world.objects == null) return;

            // Draw all objects
            for (int i = 0; i < _world.objects.Count; i++)
            {
                var obj = _world.objects[i];
                bool isSelected = (i == _selectedObjectIndex);

                DrawObjectGizmo(obj, i, isSelected);
            }

            // Draw selected object handle
            if (_selectedObjectIndex >= 0 && _selectedObjectIndex < _world.objects.Count)
            {
                DrawObjectHandle(_selectedObjectIndex);
            }
        }

        private void DrawObjectGizmo(SdfObjectDefinition obj, int index, bool isSelected)
        {
            Color gizmoColor = isSelected ? Color.yellow : GetShapeColor(obj.shapeType);
            gizmoColor.a = isSelected ? 0.8f : 0.4f;

            Handles.color = gizmoColor;

            switch (obj.shapeType)
            {
                case SdfShapeType.Sphere:
                case SdfShapeType.Metaball:
                    Handles.DrawWireDisc(obj.position, Vector3.up, obj.radius);
                    Handles.DrawWireDisc(obj.position, Vector3.forward, obj.radius);
                    Handles.DrawWireDisc(obj.position, Vector3.right, obj.radius);
                    break;

                case SdfShapeType.Box:
                    Vector3 size = obj.halfExtents != Vector3.zero ? obj.halfExtents : Vector3.one * obj.radius * 2;
                    Handles.DrawWireCube(obj.position, size);
                    break;

                case SdfShapeType.Torus:
                    Handles.DrawWireDisc(obj.position, Vector3.up, obj.majorRadius);
                    Handles.DrawWireDisc(obj.position, Vector3.up, obj.majorRadius - obj.minorRadius);
                    Handles.DrawWireDisc(obj.position, Vector3.up, obj.majorRadius + obj.minorRadius);
                    break;

                case SdfShapeType.Cylinder:
                case SdfShapeType.Capsule:
                    Handles.DrawWireDisc(obj.position + Vector3.up * obj.radius, Vector3.up, obj.radius * 0.5f);
                    Handles.DrawWireDisc(obj.position - Vector3.up * obj.radius, Vector3.up, obj.radius * 0.5f);
                    break;
            }

            // Label
            if (isSelected)
            {
                Handles.Label(obj.position + Vector3.up * (obj.radius + 0.5f),
                    $"{GetShapeIcon(obj.shapeType)} {obj.name}",
                    EditorStyles.boldLabel);
            }
        }

        private void DrawObjectHandle(int index)
        {
            var obj = _world.objects[index];

            EditorGUI.BeginChangeCheck();

            // Position Handle
            Vector3 newPosition = Handles.PositionHandle(obj.position, Quaternion.identity);

            if (EditorGUI.EndChangeCheck())
            {
                Undo.RecordObject(_world, "Move SDF Object");
                obj.position = newPosition;
                _world.objects[index] = obj;
                EditorUtility.SetDirty(_world);
            }

            // Radius Handle (for spheres, metaballs)
            if (obj.shapeType == SdfShapeType.Sphere || obj.shapeType == SdfShapeType.Metaball)
            {
                EditorGUI.BeginChangeCheck();
                float newRadius = Handles.RadiusHandle(Quaternion.identity, obj.position, obj.radius);

                if (EditorGUI.EndChangeCheck())
                {
                    Undo.RecordObject(_world, "Resize SDF Object");
                    obj.radius = Mathf.Max(0.1f, newRadius);
                    _world.objects[index] = obj;
                    EditorUtility.SetDirty(_world);
                }
            }
        }

        private Color GetShapeColor(SdfShapeType shapeType)
        {
            return shapeType switch
            {
                SdfShapeType.Sphere => new Color(0.3f, 0.7f, 1f),
                SdfShapeType.Box => new Color(1f, 0.5f, 0.3f),
                SdfShapeType.Torus => new Color(0.5f, 1f, 0.5f),
                SdfShapeType.Metaball => new Color(1f, 0.3f, 0.8f),
                SdfShapeType.Cylinder => new Color(0.8f, 0.8f, 0.3f),
                SdfShapeType.Capsule => new Color(0.6f, 0.4f, 1f),
                _ => Color.white
            };
        }

        // =====================================================================
        // Object Creation
        // =====================================================================

        private void AddObject(SdfShapeType shapeType, string baseName)
        {
            var obj = new SdfObjectDefinition
            {
                name = $"{baseName} {_world.objects.Count + 1}",
                shapeType = shapeType,
                position = GetSpawnPosition(),
                radius = Random.Range(1f, 2.5f)
            };

            if (shapeType == SdfShapeType.Torus)
            {
                obj.majorRadius = obj.radius;
                obj.minorRadius = obj.radius * 0.3f;
            }
            else if (shapeType == SdfShapeType.Box)
            {
                obj.halfExtents = Vector3.one * obj.radius;
            }

            Undo.RecordObject(_world, $"Add {baseName}");
            _world.objects.Add(obj);
            _selectedObjectIndex = _world.objects.Count - 1;
            EditorUtility.SetDirty(_world);
            serializedObject.Update();
        }

        private Vector3 GetSpawnPosition()
        {
            // Try to spawn in view
            if (SceneView.lastActiveSceneView != null)
            {
                var cam = SceneView.lastActiveSceneView.camera;
                if (cam != null)
                {
                    return cam.transform.position + cam.transform.forward * 10f;
                }
            }
            return Random.insideUnitSphere * 5f;
        }

        // =====================================================================
        // Preset Creation
        // =====================================================================

        private void CreatePlanetPreset()
        {
            if (!ConfirmPreset("Planet")) return;

            Undo.RecordObject(_world, "Create Planet Preset");
            _world.objects.Clear();

            // Sun
            _world.objects.Add(new SdfObjectDefinition
            {
                name = "Sun",
                shapeType = SdfShapeType.Sphere,
                position = Vector3.zero,
                radius = 5f
            });

            // Planet
            _world.objects.Add(new SdfObjectDefinition
            {
                name = "Planet",
                shapeType = SdfShapeType.Sphere,
                position = new Vector3(15f, 0f, 0f),
                radius = 2f
            });

            // Moon
            _world.objects.Add(new SdfObjectDefinition
            {
                name = "Moon",
                shapeType = SdfShapeType.Sphere,
                position = new Vector3(18f, 1f, 0f),
                radius = 0.5f
            });

            _world.globalSmoothness = 0.5f;
            FinalizePreset();
        }

        private void CreateCosmicPreset()
        {
            if (!ConfirmPreset("Cosmic")) return;

            Undo.RecordObject(_world, "Create Cosmic Preset");
            _world.objects.Clear();

            // Central sun
            _world.objects.Add(new SdfObjectDefinition
            {
                name = "Central Star",
                shapeType = SdfShapeType.Sphere,
                position = Vector3.zero,
                radius = 8f
            });

            // Orbiting planets
            for (int i = 0; i < 5; i++)
            {
                float angle = i * Mathf.PI * 2f / 5f;
                float distance = 15f + i * 5f;
                _world.objects.Add(new SdfObjectDefinition
                {
                    name = $"Planet {i + 1}",
                    shapeType = SdfShapeType.Sphere,
                    position = new Vector3(Mathf.Cos(angle) * distance, 0, Mathf.Sin(angle) * distance),
                    radius = 1f + Random.Range(0f, 1.5f)
                });
            }

            // Ring (torus)
            _world.objects.Add(new SdfObjectDefinition
            {
                name = "Asteroid Belt",
                shapeType = SdfShapeType.Torus,
                position = Vector3.zero,
                majorRadius = 25f,
                minorRadius = 0.5f
            });

            _world.globalSmoothness = 1f;
            FinalizePreset();
        }

        private void CreateAbstractPreset()
        {
            if (!ConfirmPreset("Abstract")) return;

            Undo.RecordObject(_world, "Create Abstract Preset");
            _world.objects.Clear();

            // Central torus
            _world.objects.Add(new SdfObjectDefinition
            {
                name = "Core Torus",
                shapeType = SdfShapeType.Torus,
                position = Vector3.zero,
                majorRadius = 5f,
                minorRadius = 1f
            });

            // Orbiting spheres
            for (int i = 0; i < 8; i++)
            {
                float angle = i * Mathf.PI * 2f / 8f;
                _world.objects.Add(new SdfObjectDefinition
                {
                    name = $"Orb {i + 1}",
                    shapeType = SdfShapeType.Sphere,
                    position = new Vector3(
                        Mathf.Cos(angle) * 8f,
                        Mathf.Sin(angle * 2f) * 3f,
                        Mathf.Sin(angle) * 8f
                    ),
                    radius = 1.5f
                });
            }

            _world.globalSmoothness = 2f;
            FinalizePreset();
        }

        private void CreateTerrainPreset()
        {
            if (!ConfirmPreset("Terrain")) return;

            Undo.RecordObject(_world, "Create Terrain Preset");
            _world.objects.Clear();

            // Ground plane
            _world.groundEnabled = true;
            _world.groundHeight = 0f;

            // Hills (large spheres below ground)
            for (int i = 0; i < 6; i++)
            {
                Vector3 pos = Random.insideUnitSphere * 20f;
                pos.y = -5f + Random.Range(-3f, 3f);
                _world.objects.Add(new SdfObjectDefinition
                {
                    name = $"Hill {i + 1}",
                    shapeType = SdfShapeType.Sphere,
                    position = pos,
                    radius = Random.Range(8f, 15f)
                });
            }

            // Rocks (small boxes)
            for (int i = 0; i < 5; i++)
            {
                Vector3 pos = Random.insideUnitSphere * 15f;
                pos.y = Random.Range(0.5f, 2f);
                _world.objects.Add(new SdfObjectDefinition
                {
                    name = $"Rock {i + 1}",
                    shapeType = SdfShapeType.Box,
                    position = pos,
                    halfExtents = Vector3.one * Random.Range(1f, 3f)
                });
            }

            _world.globalSmoothness = 1.5f;
            FinalizePreset();
        }

        private void CreateMetaballPreset()
        {
            if (!ConfirmPreset("Metaballs")) return;

            Undo.RecordObject(_world, "Create Metaballs Preset");
            _world.objects.Clear();

            // Cluster of metaballs
            for (int i = 0; i < 10; i++)
            {
                _world.objects.Add(new SdfObjectDefinition
                {
                    name = $"Metaball {i + 1}",
                    shapeType = SdfShapeType.Metaball,
                    position = Random.insideUnitSphere * 8f,
                    radius = Random.Range(1.5f, 3f)
                });
            }

            _world.globalSmoothness = 3f; // High smoothness for metaball effect
            FinalizePreset();
        }

        private void CreateAtomPreset()
        {
            if (!ConfirmPreset("Atom")) return;

            Undo.RecordObject(_world, "Create Atom Preset");
            _world.objects.Clear();

            // Nucleus
            _world.objects.Add(new SdfObjectDefinition
            {
                name = "Nucleus",
                shapeType = SdfShapeType.Sphere,
                position = Vector3.zero,
                radius = 2f
            });

            // Electron orbits (tori)
            for (int i = 0; i < 3; i++)
            {
                Quaternion rot = Quaternion.Euler(i * 60f, i * 30f, 0);
                // Note: Torus rotation would require additional properties
                // For now, we place them at different positions
                _world.objects.Add(new SdfObjectDefinition
                {
                    name = $"Orbit {i + 1}",
                    shapeType = SdfShapeType.Torus,
                    position = Vector3.zero,
                    majorRadius = 6f + i * 2f,
                    minorRadius = 0.1f
                });
            }

            // Electrons
            for (int i = 0; i < 6; i++)
            {
                float angle = i * Mathf.PI * 2f / 6f;
                float orbit = 6f + (i % 3) * 2f;
                _world.objects.Add(new SdfObjectDefinition
                {
                    name = $"Electron {i + 1}",
                    shapeType = SdfShapeType.Sphere,
                    position = new Vector3(
                        Mathf.Cos(angle) * orbit,
                        Mathf.Sin(angle * 0.5f) * 3f,
                        Mathf.Sin(angle) * orbit
                    ),
                    radius = 0.4f
                });
            }

            _world.globalSmoothness = 0.3f;
            FinalizePreset();
        }

        private bool ConfirmPreset(string presetName)
        {
            if (_world.objects.Count > 0)
            {
                return EditorUtility.DisplayDialog(
                    $"Load {presetName} Preset",
                    "This will replace all current objects. Continue?",
                    "Yes", "Cancel");
            }
            return true;
        }

        private void FinalizePreset()
        {
            _selectedObjectIndex = -1;
            EditorUtility.SetDirty(_world);
            serializedObject.Update();
            SceneView.RepaintAll();
        }
    }
}
