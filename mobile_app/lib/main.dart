import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:file_picker/file_picker.dart';
import 'package:image_picker/image_picker.dart';
import 'package:shared_preferences/shared_preferences.dart';

const String serverUrl = ""; // <-- CHANGE THIS

void main() {
  runApp(const SoundboardApp());
}

// Add this above SoundboardApp class
class ServerUrlManager {
  static String _serverUrl = serverUrl;
  static String get url => _serverUrl;
  static set url(String value) {
    _serverUrl = _fixUrl(value);
    _saveUrl(_serverUrl);
  }

  static String _fixUrl(String value) {
    var v = value.trim();
    if (v.isEmpty) return "";
    if (!v.startsWith('http://') && !v.startsWith('https://')) {
      v = 'http://$v';
    }
    if (v.endsWith('/')) v = v.substring(0, v.length - 1);
    return v;
  }

  static Future<void> _saveUrl(String url) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('server_url', url);
  }

  static Future<void> loadUrl() async {
    final prefs = await SharedPreferences.getInstance();
    final saved = prefs.getString('server_url');
    if (saved != null && saved.isNotEmpty) {
      _serverUrl = saved;
    }
  }
}

// Add this function to fetch settings from your backend (adjust endpoint as needed)
Future<Map<String, dynamic>> fetchSettings() async {
  if (ServerUrlManager.url.isEmpty) return {}; // Don't try to fetch if no URL
  try {
    final res = await http.get(Uri.parse('${ServerUrlManager.url}/settings'));
    if (res.statusCode == 200) {
      return json.decode(res.body) as Map<String, dynamic>;
    }
  } catch (_) {
    // Ignore errors, just return empty
  }
  return {};
}

class SoundboardApp extends StatefulWidget {
  const SoundboardApp({super.key});

  @override
  State<SoundboardApp> createState() => _SoundboardAppState();
}

class _SoundboardAppState extends State<SoundboardApp> {
  bool darkMode = false;
  bool loadingSettings = true;

  @override
  void initState() {
    super.initState();
    _initSettings();
  }

  Future<void> _initSettings() async {
    await ServerUrlManager.loadUrl(); // <-- Load saved URL on startup
    final settings = await fetchSettings();
    setState(() {
      darkMode = settings['dark_mode'] ?? false;
      loadingSettings = false;
    });
  }

  // Send dark mode status to backend
  Future<void> _sendDarkModeStatus(bool value) async {
    if (ServerUrlManager.url.isEmpty) return;
    try {
      await http.post(
        Uri.parse('${ServerUrlManager.url}/settings'),
        headers: {'Content-Type': 'application/json'},
        body: json.encode({'dark_mode': value}),
      );
    } catch (_) {}
  }

  // Add this function to show the dialog and update the server URL
  Future<void> _editServerUrl(BuildContext context) async {
    print("Edit Server URL dialog opened"); // Debug line
    final controller = TextEditingController(text: ServerUrlManager.url);
    final result = await showDialog<String>(
      context: context,
      builder:
          (ctx) => AlertDialog(
            title: const Text('Edit Server URL'),
            content: TextField(
              controller: controller,
              decoration: const InputDecoration(labelText: 'Server URL'),
            ),
            actions: [
              TextButton(
                onPressed: () => Navigator.pop(ctx),
                child: const Text('Cancel'),
              ),
              TextButton(
                onPressed: () => Navigator.pop(ctx, controller.text.trim()),
                child: const Text('Save'),
              ),
            ],
          ),
    );
    if (result != null && result.isNotEmpty) {
      setState(() {
        ServerUrlManager.url = result;
        loadingSettings = true;
      });
      await _initSettings();
    }
  }

  void _toggleDarkMode() async {
    setState(() => darkMode = !darkMode);
    await _sendDarkModeStatus(darkMode);
  }

  @override
  Widget build(BuildContext context) {
    final brightness = darkMode ? Brightness.dark : Brightness.light;
    return MaterialApp(
      title: 'Soundboard',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: Colors.blue,
          brightness: brightness,
        ),
        useMaterial3: true,
        brightness: brightness,
      ),
      home: SoundboardHomePage(
        darkMode: darkMode,
        onToggleDarkMode: _toggleDarkMode,
        onEditServerUrl: (ctx) => _editServerUrl(ctx),
      ),
    );
  }
}

// Update SoundboardHomePage to accept onEditServerUrl
class SoundboardHomePage extends StatefulWidget {
  final bool darkMode;
  final VoidCallback onToggleDarkMode;
  final void Function(BuildContext) onEditServerUrl; // <-- change here
  const SoundboardHomePage({
    super.key,
    required this.darkMode,
    required this.onToggleDarkMode,
    required this.onEditServerUrl,
  });

  @override
  State<SoundboardHomePage> createState() => _SoundboardHomePageState();
}

class _SoundboardHomePageState extends State<SoundboardHomePage> {
  List<String> sounds = [];
  List<String> favourites = [];
  List<String> soundOrder = [];
  bool loading = true;

  @override
  void initState() {
    super.initState();
    fetchAll();
  }

  Future<void> fetchAll() async {
    setState(() => loading = true);
    if (ServerUrlManager.url.isEmpty) {
      setState(() {
        sounds = [];
        favourites = [];
        soundOrder = [];
        loading = false;
      });
      return;
    }
    try {
      final sRes = await http.get(Uri.parse('${ServerUrlManager.url}/sounds'));
      final fRes = await http.get(
        Uri.parse('${ServerUrlManager.url}/favourites'),
      );
      final oRes = await http.get(Uri.parse('${ServerUrlManager.url}/order'));
      setState(() {
        sounds = List<String>.from(json.decode(sRes.body));
        favourites = List<String>.from(json.decode(fRes.body));
        soundOrder = List<String>.from(json.decode(oRes.body));
        loading = false;
      });
    } catch (_) {
      setState(() {
        sounds = [];
        favourites = [];
        soundOrder = [];
        loading = false;
      });
    }
  }

  Future<void> reloadAll() async {
    final settings = await fetchSettings();
    if (mounted) {
      if (settings.containsKey('dark_mode')) {
        // Only toggle if the value is different
        if (settings['dark_mode'] != widget.darkMode) {
          widget.onToggleDarkMode();
        }
      }
      await fetchAll();
    }
  }

  Future<void> playSound(String name) async {
    if (ServerUrlManager.url.isEmpty) return;
    try {
      await http.post(
        Uri.parse('${ServerUrlManager.url}/play'),
        headers: {'Content-Type': 'application/json'},
        body: json.encode({'name': name}),
      );
    } catch (_) {}
  }

  Future<void> toggleFavourite(String name) async {
    setState(() {
      if (favourites.contains(name)) {
        favourites.remove(name);
      } else {
        favourites.add(name);
      }
    });
    if (ServerUrlManager.url.isEmpty) return;
    try {
      await http.post(
        Uri.parse('${ServerUrlManager.url}/favourites'),
        headers: {'Content-Type': 'application/json'},
        body: json.encode(favourites),
      );
    } catch (_) {}
  }

  Future<void> deleteSound(String name) async {
    if (ServerUrlManager.url.isEmpty) return;
    try {
      await http.delete(Uri.parse('${ServerUrlManager.url}/delete/$name'));
      await fetchAll();
    } catch (_) {}
  }

  Future<void> stopAll() async {
    if (ServerUrlManager.url.isEmpty) return;
    try {
      await http.post(Uri.parse('${ServerUrlManager.url}/stop'));
    } catch (_) {}
  }

  Future<void> uploadSound() async {
    FilePickerResult? result = await FilePicker.platform.pickFiles(
      type: FileType.audio,
    );
    if (result == null) return;
    final audioFile = File(result.files.single.path!);

    final nameController = TextEditingController(
      text: audioFile.path
          .split(Platform.pathSeparator)
          .last
          .replaceAll(RegExp(r'\.[^\.]+$'), ''),
    );
    XFile? imageFile;

    await showDialog(
      context: context,
      builder:
          (ctx) => AlertDialog(
            title: const Text("Upload Sound"),
            content: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                TextField(
                  controller: nameController,
                  decoration: const InputDecoration(labelText: "Sound Name"),
                ),
                const SizedBox(height: 10),
                ElevatedButton.icon(
                  icon: const Icon(Icons.image),
                  label: Text(
                    imageFile == null
                        ? "Select Icon (Optional)"
                        : "Change Icon",
                  ),
                  onPressed: () async {
                    final picked = await ImagePicker().pickImage(
                      source: ImageSource.gallery,
                    );
                    if (picked != null) {
                      setState(() => imageFile = picked);
                      Navigator.of(ctx).pop();
                      await uploadSound(); // re-open dialog with image selected
                    }
                  },
                ),
              ],
            ),
            actions: [
              TextButton(
                onPressed: () => Navigator.pop(ctx),
                child: const Text("Cancel"),
              ),
              TextButton(
                onPressed: () async {
                  final name = nameController.text.trim();
                  if (name.isEmpty || ServerUrlManager.url.isEmpty) return;
                  final ext = audioFile.path.split('.').last;
                  final renamed = await audioFile.copy(
                    '${audioFile.parent.path}/$name.$ext',
                  );
                  final req = http.MultipartRequest(
                    'POST',
                    Uri.parse('${ServerUrlManager.url}/upload'),
                  );
                  req.files.add(
                    await http.MultipartFile.fromPath('file', renamed.path),
                  );
                  if (imageFile != null) {
                    req.files.add(
                      await http.MultipartFile.fromPath(
                        'image',
                        imageFile!.path,
                        filename: "$name.png",
                      ),
                    );
                  }
                  try {
                    await req.send();
                  } catch (_) {}
                  Navigator.pop(ctx);
                  await fetchAll();
                },
                child: const Text("Upload"),
              ),
            ],
          ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final sortedSounds = [...sounds];
    sortedSounds.sort((a, b) {
      final aIdx = soundOrder.indexOf(a);
      final bIdx = soundOrder.indexOf(b);
      if (aIdx != -1 && bIdx != -1) return aIdx - bIdx;
      if (aIdx != -1) return -1;
      if (bIdx != -1) return 1;
      return a.compareTo(b);
    });

    final size = MediaQuery.of(context).size;
    final isPortrait = size.height > size.width;
    // 2 columns in portrait, 5 rows; 5 columns in landscape, 2 rows
    final crossAxisCount = isPortrait ? 2 : 5;
    final mainAxisCount = isPortrait ? 5 : 2;
    // Calculate tile size to fit perfectly
    final tileWidth =
        (size.width - 24 - (crossAxisCount - 1) * 18) / crossAxisCount;
    final tileHeight =
        (size.height -
            MediaQuery.of(context).padding.top -
            (isPortrait ? 54 : 0) -
            24 -
            (mainAxisCount - 1) * 18) /
        mainAxisCount;
    final childAspectRatio = tileWidth / tileHeight;

    // Add more horizontal padding in landscape
    final gridPadding =
        isPortrait
            ? const EdgeInsets.all(15)
            : const EdgeInsets.fromLTRB(15, 35, 30, 15);

    return Scaffold(
      body:
          loading
              ? const Center(child: CircularProgressIndicator())
              : CustomScrollView(
                slivers: [
                  if (isPortrait)
                    SliverAppBar(
                      title: const Text('Soundboard'),
                      floating: true,
                      snap: true,
                      pinned: false,
                      forceElevated: true,
                      toolbarHeight: 54,
                      elevation: 2,
                      backgroundColor: Theme.of(context).colorScheme.surface,
                      actions: [
                        IconButton(
                          icon: const Icon(Icons.refresh),
                          onPressed: reloadAll,
                        ),
                        IconButton(
                          icon: const Icon(Icons.upload),
                          onPressed: uploadSound,
                          tooltip: "Upload",
                        ),
                        IconButton(
                          icon: Icon(
                            widget.darkMode
                                ? Icons.dark_mode
                                : Icons.light_mode,
                          ),
                          onPressed: widget.onToggleDarkMode,
                        ),
                        IconButton(
                          icon: const Icon(Icons.settings_ethernet),
                          tooltip: "Edit Server URL",
                          onPressed: () {
                            widget.onEditServerUrl(
                              context,
                            ); // <-- pass context here
                          },
                        ),
                      ],
                    ),
                  SliverPadding(
                    padding: gridPadding,
                    sliver: SliverGrid(
                      delegate: SliverChildBuilderDelegate((context, idx) {
                        final name = sortedSounds[idx];
                        return _buildSoundCard(name, key: ValueKey(name));
                      }, childCount: sortedSounds.length),
                      gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
                        crossAxisCount: crossAxisCount,
                        childAspectRatio: childAspectRatio,
                        crossAxisSpacing: 18,
                        mainAxisSpacing: 18,
                      ),
                    ),
                  ),
                ],
              ),
      floatingActionButton: FloatingActionButton(
        onPressed: stopAll,
        tooltip: "Stop All",
        backgroundColor: Colors.red,
        child: const Icon(Icons.stop, color: Colors.white),
      ),
    );
  }

  Widget _buildSoundCard(String name, {Key? key}) {
    final baseName = name.replaceAll(RegExp(r'\.[^\.]+$'), '');
    final isFav = favourites.contains(name);
    final imageUrl = '${ServerUrlManager.url}/sounds/$baseName.png';

    return Card(
      key: key,
      elevation: 2,
      margin: const EdgeInsets.all(4),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
      child: InkWell(
        borderRadius: BorderRadius.circular(14),
        onTap: () => playSound(name),
        child: Stack(
          children: [
            // Background image
            ClipRRect(
              borderRadius: BorderRadius.circular(14),
              child: FutureBuilder(
                future: http.get(Uri.parse(imageUrl)),
                builder: (context, snapshot) {
                  if (snapshot.connectionState == ConnectionState.done &&
                      snapshot.hasData &&
                      (snapshot.data as http.Response).statusCode == 200) {
                    return Image.network(
                      imageUrl,
                      width: double.infinity,
                      height: double.infinity,
                      fit: BoxFit.cover,
                    );
                  }
                  // Placeholder background
                  return Container(
                    width: double.infinity,
                    height: double.infinity,
                    color: Colors.blueGrey[200],
                  );
                },
              ),
            ),
            // Overlay for readability
            Container(
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(14),
                color: Colors.black.withOpacity(0.18),
              ),
            ),
            // Content
            Padding(
              padding: const EdgeInsets.symmetric(
                horizontal: 8.0,
                vertical: 8.0,
              ),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  // Top row: fav and delete
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      // Favourite button - move half out of the top left corner
                      Transform.translate(
                        offset: const Offset(-12, -12),
                        child: IconButton(
                          icon: Icon(
                            isFav ? Icons.favorite : Icons.favorite_border,
                            color: isFav ? Colors.red : Colors.white,
                            size: 18,
                            shadows: const [
                              Shadow(blurRadius: 4, color: Colors.black),
                            ],
                          ),
                          padding: EdgeInsets.zero,
                          constraints: const BoxConstraints(
                            minWidth: 24,
                            minHeight: 24,
                          ),
                          onPressed: () => toggleFavourite(name),
                        ),
                      ),
                      // Delete button - move half out of the top right corner
                      Transform.translate(
                        offset: const Offset(12, -12),
                        child: IconButton(
                          icon: const Icon(
                            Icons.delete,
                            color: Colors.white,
                            size: 18,
                            shadows: [
                              Shadow(blurRadius: 4, color: Colors.black),
                            ],
                          ),
                          padding: EdgeInsets.zero,
                          constraints: const BoxConstraints(
                            minWidth: 24,
                            minHeight: 24,
                          ),
                          onPressed: () async {
                            final confirm = await showDialog<bool>(
                              context: context,
                              builder:
                                  (ctx) => AlertDialog(
                                    title: const Text("Delete Sound"),
                                    content: Text("Delete \"$baseName\"?"),
                                    actions: [
                                      TextButton(
                                        onPressed:
                                            () => Navigator.pop(ctx, false),
                                        child: const Text("Cancel"),
                                      ),
                                      TextButton(
                                        onPressed:
                                            () => Navigator.pop(ctx, true),
                                        child: const Text("Delete"),
                                      ),
                                    ],
                                  ),
                            );
                            if (confirm == true) {
                              await deleteSound(name);
                            }
                          },
                        ),
                      ),
                    ],
                  ),
                  // Center: name
                  Expanded(
                    child: Center(
                      child: Text(
                        baseName,
                        style: const TextStyle(
                          fontWeight: FontWeight.bold,
                          fontSize: 18,
                          color: Colors.white,
                          shadows: [
                            Shadow(
                              blurRadius: 6,
                              color: Colors.black54,
                              offset: Offset(0, 1),
                            ),
                          ],
                        ),
                        textAlign: TextAlign.center,
                        maxLines: 2,
                        overflow: TextOverflow.ellipsis,
                      ),
                    ),
                  ),
                  // Bottom: empty for spacing
                  const SizedBox(height: 2),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
