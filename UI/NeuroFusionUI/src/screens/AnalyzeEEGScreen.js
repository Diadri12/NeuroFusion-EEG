import React, { useState, useEffect, useRef } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  StatusBar,
  SafeAreaView,
  Animated,
  ScrollView,
} from 'react-native';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import * as DocumentPicker from 'expo-document-picker';
import InvalidFormatDialog from './InvalidFormatDialog';
import { useRouter } from 'expo-router';

const FILE_TYPES = [
  { type: '.CSV',  color: '#5A189A' },
  { type: '.EDF',  color: '#6A1B9A' },
  { type: '.TXT',  color: '#7B2CBF' },
  { type: '.MAT',  color: '#9D4EDD' },
  { type: '.XLSX', color: '#B983FF' },
  { type: '.XLS',  color: '#D0BFFF' },
];

const RECENT_FILES = [
  { name: 'eeg_session_12mar.csv', type: '.CSV', size: '2.3 MB', date: '12 Mar' },
  { name: 'patient_eeg_data.edf',  type: '.EDF', size: '5.1 MB', date: '10 Mar' },
];

const AnalyzeEEGScreen = ({ onStartAnalysis, onGoBack }) => {
  const router = useRouter();
  const [selectedFile,      setSelectedFile]      = useState(null);
  const [selectedFileType,  setSelectedFileType]  = useState('');
  const [showInvalidDialog, setShowInvalidDialog] = useState(false);
  const [showRecent,        setShowRecent]        = useState(false);

  // ── Animation values ──
  const fadeAnim      = useRef(new Animated.Value(0)).current;
  const slideAnim     = useRef(new Animated.Value(20)).current;
  const scaleAnim     = useRef(new Animated.Value(0.9)).current;
  const uploadBoxAnim = useRef(new Animated.Value(0)).current;
  const buttonFade    = useRef(new Animated.Value(0)).current;
  const pulseAnim     = useRef(new Animated.Value(1)).current;
  const checkAnim     = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.parallel([
      Animated.timing(fadeAnim,  { toValue: 1, duration: 500, useNativeDriver: true }),
      Animated.spring(slideAnim, { toValue: 0, friction: 8, tension: 40, useNativeDriver: true }),
      Animated.spring(scaleAnim, { toValue: 1, friction: 8, tension: 40, useNativeDriver: true }),
    ]).start();

    Animated.sequence([
      Animated.delay(200),
      Animated.spring(uploadBoxAnim, { toValue: 1, friction: 8, tension: 40, useNativeDriver: true }),
    ]).start();

    Animated.sequence([
      Animated.delay(400),
      Animated.timing(buttonFade, { toValue: 1, duration: 500, useNativeDriver: true }),
    ]).start();

    if (!selectedFile) {
      Animated.loop(
        Animated.sequence([
          Animated.timing(pulseAnim, { toValue: 1.02, duration: 2000, useNativeDriver: true }),
          Animated.timing(pulseAnim, { toValue: 1,    duration: 2000, useNativeDriver: true }),
        ])
      ).start();
    }
  }, [selectedFile]);

  // Animate checkmark when file selected
  useEffect(() => {
    if (selectedFile) {
      Animated.spring(checkAnim, { toValue: 1, friction: 5, tension: 40, useNativeDriver: true }).start();
    } else {
      checkAnim.setValue(0);
    }
  }, [selectedFile]);

  const pickDocument = async () => {
    try {
      const result = await DocumentPicker.getDocumentAsync({
        type: '*/*',
        copyToCacheDirectory: true,
      });

      if (!result.canceled) {
        const file      = result.assets[0];
        const extension = '.' + file.name.split('.').pop().toUpperCase();
        const supported = FILE_TYPES.map(ft => ft.type);

        if (supported.includes(extension)) {
          setSelectedFile(file);
          setSelectedFileType(extension);

          Animated.sequence([
            Animated.timing(uploadBoxAnim, { toValue: 0.95, duration: 100, useNativeDriver: true }),
            Animated.spring(uploadBoxAnim, { toValue: 1,    friction: 5,   useNativeDriver: true }),
          ]).start();
        } else {
          setShowInvalidDialog(true);
        }
      }
    } catch (err) {
      console.error('Error picking document:', err);
    }
  };

  const handleAnalyze = () => {
    if (!selectedFile) return;
    router.push({
      pathname: '/analyzing',
      params: { fileUri: selectedFile.uri, fileType: selectedFileType },
    });
  };

  const handleRemoveFile = () => {
    setSelectedFile(null);
    setSelectedFileType('');
  };

  const formatFileSize = (bytes) => {
    if (!bytes)              return 'Unknown size';
    if (bytes < 1024)        return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const getFileTypeColor = (type) => {
    return FILE_TYPES.find(ft => ft.type === type)?.color || '#B844FF';
  };

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="dark-content" backgroundColor="#FFFFFF" />

      {/* Header */}
      <Animated.View style={[styles.header, { opacity: fadeAnim, transform: [{ translateY: slideAnim }] }]}>
        <TouchableOpacity onPress={onGoBack} style={styles.backButton}>
          <MaterialCommunityIcons name="arrow-left" size={28} color="#333333" />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Analyze EEG Data</Text>
        <View style={{ width: 28 }} />
      </Animated.View>

      <ScrollView style={styles.scroll} showsVerticalScrollIndicator={false}>
        <View style={styles.content}>

          {/* ── Upload Area ── */}
          <Animated.View style={{
            opacity: uploadBoxAnim,
            transform: [{ scale: !selectedFile ? pulseAnim : uploadBoxAnim }],
          }}>
            <TouchableOpacity
              style={[styles.uploadArea, selectedFile && styles.uploadAreaFilled]}
              onPress={pickDocument}
              activeOpacity={0.7}
            >
              {/* Icon */}
              <Animated.View style={{
                transform: [{ scale: selectedFile ? checkAnim : new Animated.Value(1) }],
              }}>
                <View style={[styles.iconCircle, selectedFile && styles.iconCircleFilled]}>
                  <MaterialCommunityIcons
                    name={selectedFile ? 'check' : 'folder-outline'}
                    size={40}
                    color={selectedFile ? '#FFF' : '#B844FF'}
                  />
                </View>
              </Animated.View>

              <Text style={styles.uploadTitle}>
                {selectedFile ? selectedFile.name : 'Select a File'}
              </Text>
              <Text style={styles.uploadSubtitle}>
                {selectedFile ? 'Tap to change file' : 'Browse and choose'}
              </Text>

              {/* File details when selected */}
              {selectedFile && (
                <View style={styles.fileDetailsCard}>
                  <View style={styles.fileDetailRow}>
                    <View style={[styles.typeTag, { backgroundColor: getFileTypeColor(selectedFileType) }]}>
                      <Text style={styles.typeTagText}>{selectedFileType}</Text>
                    </View>
                    <Text style={styles.fileSizeText}>
                      {formatFileSize(selectedFile.size)}
                    </Text>
                  </View>
                  <View style={styles.fileDetailRow}>
                    <MaterialCommunityIcons name="shield-check" size={16} color="#4CAF50" />
                    <Text style={styles.fileValidText}>Valid format — ready to analyze</Text>
                  </View>
                </View>
              )}

              {/* Supported formats when no file */}
              {!selectedFile && (
                <View style={styles.supportedTypesContainer}>
                  <Text style={styles.supportedTypesTitle}>Supported formats:</Text>
                  <View style={styles.supportedTypesList}>
                    {FILE_TYPES.map(ft => (
                      <View key={ft.type} style={styles.supportedTypeItem}>
                        <View style={[styles.typeDot, { backgroundColor: ft.color }]} />
                        <Text style={styles.supportedTypeText}>{ft.type}</Text>
                      </View>
                    ))}
                  </View>
                </View>
              )}
            </TouchableOpacity>
          </Animated.View>

          {/* ── Remove file button ── */}
          {selectedFile && (
            <Animated.View style={{ opacity: buttonFade }}>
              <TouchableOpacity style={styles.removeBtn} onPress={handleRemoveFile} activeOpacity={0.7}>
                <MaterialCommunityIcons name="close-circle-outline" size={16} color="#E63946" />
                <Text style={styles.removeBtnText}>Remove file</Text>
              </TouchableOpacity>
            </Animated.View>
          )}

          {/* ── Recent Files ── */}
          <Animated.View style={[styles.recentSection, { opacity: buttonFade }]}>
            <TouchableOpacity
              style={styles.recentHeader}
              onPress={() => setShowRecent(prev => !prev)}
              activeOpacity={0.75}
            >
              <View style={styles.recentHeaderLeft}>
                <MaterialCommunityIcons name="history" size={18} color="#B844FF" />
                <Text style={styles.recentTitle}>Recent Files</Text>
              </View>
              <MaterialCommunityIcons
                name={showRecent ? 'chevron-up' : 'chevron-down'}
                size={18}
                color="#B844FF"
              />
            </TouchableOpacity>

            {showRecent && (
              <View style={styles.recentList}>
                {RECENT_FILES.map((file, index) => (
                  <TouchableOpacity
                    key={index}
                    style={styles.recentItem}
                    onPress={() => {
                      setSelectedFile({ name: file.name, size: null, uri: '' });
                      setSelectedFileType(file.type);
                      setShowRecent(false);
                    }}
                    activeOpacity={0.75}
                  >
                    <View style={[styles.recentIcon, { backgroundColor: getFileTypeColor(file.type) + '20' }]}>
                      <MaterialCommunityIcons name="file-document" size={18} color={getFileTypeColor(file.type)} />
                    </View>
                    <View style={styles.recentInfo}>
                      <Text style={styles.recentName} numberOfLines={1}>{file.name}</Text>
                      <Text style={styles.recentMeta}>{file.size} · {file.date}</Text>
                    </View>
                    <View style={[styles.typeTagSmall, { backgroundColor: getFileTypeColor(file.type) }]}>
                      <Text style={styles.typeTagSmallText}>{file.type}</Text>
                    </View>
                  </TouchableOpacity>
                ))}
              </View>
            )}
          </Animated.View>

          {/* ── Analyze Button ── */}
          <Animated.View style={{ opacity: buttonFade, transform: [{ scale: scaleAnim }] }}>
            <TouchableOpacity
              style={[styles.analyzeButton, !selectedFile && styles.analyzeButtonDisabled]}
              onPress={handleAnalyze}
              activeOpacity={0.8}
              disabled={!selectedFile}
            >
              <MaterialCommunityIcons name="play-circle" size={24} color="#FFFFFF" style={{ marginRight: 8 }} />
              <Text style={styles.analyzeButtonText}>Start Analysis</Text>
            </TouchableOpacity>
          </Animated.View>

          {/* ── Help text ── */}
          <Animated.View style={[styles.helpContainer, { opacity: buttonFade }]}>
            <MaterialCommunityIcons name="help-circle-outline" size={18} color="#999" />
            <Text style={styles.helpText}>
              Upload your EEG data file to begin seizure detection analysis
            </Text>
          </Animated.View>

        </View>
      </ScrollView>

      <InvalidFormatDialog
        visible={showInvalidDialog}
        onClose={() => setShowInvalidDialog(false)}
      />
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#F8F9FA' },
  scroll:    { flex: 1 },

  // ── Header ──
  header: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    padding: 16, backgroundColor: '#FFFFFF',
    borderBottomWidth: 1, borderBottomColor: '#E0E0E0',
  },
  backButton:  { padding: 4 },
  headerTitle: { fontSize: 20, fontWeight: 'bold', color: '#333333' },

  content: { flex: 1, padding: 24 },

  // ── Upload area ──
  uploadArea: {
    borderWidth: 2, borderColor: '#B844FF', borderStyle: 'dashed',
    borderRadius: 20, padding: 40, alignItems: 'center',
    marginBottom: 12, backgroundColor: '#FFFFFF',
  },
  uploadAreaFilled: {
    borderStyle: 'solid', borderColor: '#4CAF50',
    backgroundColor: '#F0FFF4',
  },
  iconCircle: {
    width: 80, height: 80, borderRadius: 40,
    backgroundColor: '#F5F0FF',
    alignItems: 'center', justifyContent: 'center',
    marginBottom: 16,
  },
  iconCircleFilled: { backgroundColor: '#4CAF50' },
  uploadTitle:    { fontSize: 18, fontWeight: '600', color: '#333333', marginBottom: 8, textAlign: 'center' },
  uploadSubtitle: { fontSize: 14, color: '#666666', textAlign: 'center', marginBottom: 4 },

  // ── File details ──
  fileDetailsCard: {
    marginTop: 16, paddingTop: 16,
    borderTopWidth: 1, borderTopColor: '#E0E0E0',
    width: '100%', gap: 10,
  },
  fileDetailRow: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  typeTag:       { borderRadius: 8, paddingHorizontal: 10, paddingVertical: 3 },
  typeTagText:   { fontSize: 11, fontWeight: '700', color: '#FFF' },
  fileSizeText:  { fontSize: 13, color: '#6B7280', fontWeight: '500' },
  fileValidText: { fontSize: 13, color: '#4CAF50', fontWeight: '600' },

  // ── Supported types ──
  supportedTypesContainer: {
    marginTop: 16, paddingTop: 20,
    borderTopWidth: 1, borderTopColor: '#E0E0E0', width: '100%',
  },
  supportedTypesTitle: { fontSize: 14, fontWeight: '600', color: '#666666', marginBottom: 12, textAlign: 'center' },
  supportedTypesList:  { flexDirection: 'row', flexWrap: 'wrap', justifyContent: 'center', gap: 12 },
  supportedTypeItem:   {
    flexDirection: 'row', alignItems: 'center',
    backgroundColor: '#F5F5F5', paddingHorizontal: 12, paddingVertical: 6, borderRadius: 12,
  },
  typeDot:           { width: 8, height: 8, borderRadius: 4, marginRight: 6 },
  supportedTypeText: { fontSize: 13, fontWeight: '600', color: '#333333' },

  // ── Remove button ──
  removeBtn: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'center',
    gap: 5, paddingVertical: 8, marginBottom: 12,
  },
  removeBtnText: { fontSize: 13, color: '#E63946', fontWeight: '600' },

  // ── Recent files ──
  recentSection: {
    backgroundColor: '#FFFFFF', borderRadius: 16, marginBottom: 20,
    borderWidth: 1, borderColor: '#E0E0E0', overflow: 'hidden',
  },
  recentHeader: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    padding: 16,
  },
  recentHeaderLeft: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  recentTitle:      { fontSize: 15, fontWeight: '700', color: '#333' },
  recentList:       { borderTopWidth: 1, borderTopColor: '#F0F0F0' },
  recentItem: {
    flexDirection: 'row', alignItems: 'center', gap: 12,
    paddingHorizontal: 16, paddingVertical: 12,
    borderBottomWidth: 1, borderBottomColor: '#F9F9F9',
  },
  recentIcon:       { width: 38, height: 38, borderRadius: 10, alignItems: 'center', justifyContent: 'center' },
  recentInfo:       { flex: 1 },
  recentName:       { fontSize: 13, fontWeight: '600', color: '#333' },
  recentMeta:       { fontSize: 11, color: '#9CA3AF', marginTop: 2 },
  typeTagSmall:     { borderRadius: 6, paddingHorizontal: 8, paddingVertical: 2 },
  typeTagSmallText: { fontSize: 10, fontWeight: '700', color: '#FFF' },

  // ── Analyze button ──
  analyzeButton: {
    backgroundColor: '#B844FF', borderRadius: 30, paddingVertical: 16,
    alignItems: 'center', flexDirection: 'row', justifyContent: 'center',
    shadowColor: '#B844FF', shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3, shadowRadius: 8, elevation: 5, marginBottom: 16,
  },
  analyzeButtonDisabled: { backgroundColor: '#D0D0D0', shadowOpacity: 0 },
  analyzeButtonText:     { color: '#FFFFFF', fontSize: 18, fontWeight: 'bold', letterSpacing: 0.5 },

  // ── Help ──
  helpContainer: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', paddingHorizontal: 16 },
  helpText:      { fontSize: 13, color: '#999999', marginLeft: 6, textAlign: 'center', flex: 1 },
});

export default AnalyzeEEGScreen;


















































