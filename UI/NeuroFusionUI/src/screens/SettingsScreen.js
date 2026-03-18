import React, { useState } from 'react';
import {
  View, Text, TouchableOpacity, StyleSheet,
  StatusBar, SafeAreaView, ScrollView, Switch, Modal,
} from 'react-native';
import { MaterialCommunityIcons } from '@expo/vector-icons';
 
// ── Custom Sign Out Dialog ────────────────────────────────────
const SignOutDialog = ({ visible, onConfirm, onCancel }) => (
  <Modal
    visible={visible}
    transparent
    animationType="fade"
    onRequestClose={onCancel}
  >
    <View style={dialog.overlay}>
      <View style={dialog.box}>
 
        {/* Icon */}
        <View style={dialog.iconWrap}>
          <MaterialCommunityIcons name="logout" size={32} color="#E74C3C" />
        </View>
 
        {/* Text */}
        <Text style={dialog.title}>Sign Out</Text>
        <Text style={dialog.message}>
          Are you sure you want to sign out of EpiGuard?
        </Text>
 
        {/* Buttons */}
        <View style={dialog.buttonRow}>
          <TouchableOpacity style={dialog.cancelBtn} onPress={onCancel}>
            <Text style={dialog.cancelText}>Cancel</Text>
          </TouchableOpacity>
          <TouchableOpacity style={dialog.confirmBtn} onPress={onConfirm}>
            <Text style={dialog.confirmText}>Sign Out</Text>
          </TouchableOpacity>
        </View>
 
      </View>
    </View>
  </Modal>
);
 
const dialog = StyleSheet.create({
  overlay: {
    flex: 1, backgroundColor: 'rgba(0,0,0,0.45)',
    justifyContent: 'center', alignItems: 'center', padding: 32,
  },
  box: {
    backgroundColor: '#FFF', borderRadius: 24, padding: 28,
    width: '100%', maxWidth: 340, alignItems: 'center',
    shadowColor: '#000', shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.2, shadowRadius: 20, elevation: 10,
  },
  iconWrap: {
    width: 64, height: 64, borderRadius: 32,
    backgroundColor: '#FFF0F0',
    justifyContent: 'center', alignItems: 'center', marginBottom: 16,
  },
  title:   { fontSize: 20, fontWeight: '800', color: '#1A1A2E', marginBottom: 8 },
  message: { fontSize: 14, color: '#888', textAlign: 'center', lineHeight: 20, marginBottom: 24 },
  buttonRow: { flexDirection: 'row', gap: 12, width: '100%' },
  cancelBtn: {
    flex: 1, paddingVertical: 13, borderRadius: 12,
    backgroundColor: '#F5F5F5', alignItems: 'center',
  },
  confirmBtn: {
    flex: 1, paddingVertical: 13, borderRadius: 12,
    backgroundColor: '#E74C3C', alignItems: 'center',
  },
  cancelText:  { fontSize: 15, fontWeight: '700', color: '#555' },
  confirmText: { fontSize: 15, fontWeight: '700', color: '#FFF' },
});
 
// ── Main Screen ───────────────────────────────────────────────
export default function SettingsScreen({ onSignOut, onNavigate }) {
  const [notifications,    setNotifications]    = useState(true);
  const [showSignOutModal, setShowSignOutModal] = useState(false);
 
  const handleConfirmSignOut = () => {
    setShowSignOutModal(false);
    onSignOut();
  };
 
  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor="#B844FF" />
 
      {/* Custom sign out dialog */}
      <SignOutDialog
        visible={showSignOutModal}
        onConfirm={handleConfirmSignOut}
        onCancel={() => setShowSignOutModal(false)}
      />
 
      <View style={styles.header}>
        <Text style={styles.headerTitle}>Settings</Text>
      </View>
 
      <ScrollView contentContainerStyle={styles.scroll} showsVerticalScrollIndicator={false}>
 
        {/* Profile shortcut */}
        <TouchableOpacity style={styles.profileCard} onPress={() => onNavigate('userInformation')}>
          <View style={styles.profileAvatar}>
            <MaterialCommunityIcons name="account" size={36} color="#B844FF" />
          </View>
          <View style={styles.profileInfo}>
            <Text style={styles.profileName}>My Account</Text>
            <Text style={styles.profileSub}>View and edit your profile →</Text>
          </View>
          <MaterialCommunityIcons name="chevron-right" size={20} color="#CCC" />
        </TouchableOpacity>
 
        {/* Preferences */}
        <Text style={styles.sectionLabel}>PREFERENCES</Text>
        <View style={styles.card}>
          <View style={styles.row}>
            <View style={styles.iconWrap}>
              <MaterialCommunityIcons name="bell-outline" size={20} color="#B844FF" />
            </View>
            <Text style={styles.rowLabel}>Notifications</Text>
            <Switch
              value={notifications}
              onValueChange={setNotifications}
              trackColor={{ false: '#E0E0E0', true: '#B844FF' }}
              thumbColor="#FFF"
            />
          </View>
 
        </View>
 
        {/* More */}
        <Text style={styles.sectionLabel}>MORE</Text>
        <View style={styles.card}>
          <TouchableOpacity style={styles.row} onPress={() => onNavigate('userInformation')}>
            <View style={styles.iconWrap}>
              <MaterialCommunityIcons name="account-circle-outline" size={20} color="#B844FF" />
            </View>
            <Text style={styles.rowLabel}>User Information</Text>
            <MaterialCommunityIcons name="chevron-right" size={20} color="#CCC" />
          </TouchableOpacity>
          <TouchableOpacity style={[styles.row, styles.rowBorder]} onPress={() => onNavigate('aboutApp')}>
            <View style={styles.iconWrap}>
              <MaterialCommunityIcons name="information-outline" size={20} color="#B844FF" />
            </View>
            <Text style={styles.rowLabel}>About App</Text>
            <MaterialCommunityIcons name="chevron-right" size={20} color="#CCC" />
          </TouchableOpacity>
        </View>
 
        {/* Sign Out */}
        <TouchableOpacity
          style={styles.signOutButton}
          onPress={() => setShowSignOutModal(true)}
        >
          <MaterialCommunityIcons name="logout" size={20} color="#FFF" />
          <Text style={styles.signOutText}>Sign Out</Text>
        </TouchableOpacity>
 
        <Text style={styles.footer}>EpiGuard v1.0.0 · IIT 2026</Text>
 
      </ScrollView>
    </SafeAreaView>
  );
}
 
const styles = StyleSheet.create({
  container:   { flex: 1, backgroundColor: '#F4F5F9' },
  header:      { backgroundColor: '#B844FF', paddingHorizontal: 20, paddingTop: 16, paddingBottom: 20 },
  headerTitle: { fontSize: 22, fontWeight: '800', color: '#FFF' },
  scroll:      { padding: 18, paddingBottom: 100 },
 
  profileCard: {
    flexDirection: 'row', alignItems: 'center', gap: 14,
    backgroundColor: '#FFF', borderRadius: 16, padding: 16, marginBottom: 22,
    shadowColor: '#000', shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.07, shadowRadius: 8, elevation: 3,
  },
  profileAvatar: {
    width: 58, height: 58, borderRadius: 29,
    backgroundColor: '#F5EEFF', justifyContent: 'center', alignItems: 'center',
  },
  profileInfo: { flex: 1 },
  profileName: { fontSize: 16, fontWeight: '700', color: '#1A1A2E' },
  profileSub:  { fontSize: 12, color: '#B844FF', marginTop: 3 },
 
  sectionLabel: {
    fontSize: 10, fontWeight: '800', color: '#BBB',
    letterSpacing: 1.2, marginBottom: 8, marginLeft: 4,
    textTransform: 'uppercase',
  },
  card: {
    backgroundColor: '#FFF', borderRadius: 16, overflow: 'hidden', marginBottom: 20,
    shadowColor: '#000', shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.06, shadowRadius: 6, elevation: 2,
  },
  row:       { flexDirection: 'row', alignItems: 'center', paddingHorizontal: 16, paddingVertical: 14, gap: 14 },
  rowBorder: { borderTopWidth: 1, borderTopColor: '#F5F5F5' },
  rowLabel:  { flex: 1, fontSize: 15, fontWeight: '600', color: '#333' },
  iconWrap:  { width: 38, height: 38, borderRadius: 10, backgroundColor: '#F5EEFF', justifyContent: 'center', alignItems: 'center' },
 
  signOutButton: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'center',
    gap: 10, backgroundColor: '#E74C3C', borderRadius: 16,
    paddingVertical: 16, marginBottom: 16,
  },
  signOutText: { fontSize: 16, fontWeight: '700', color: '#FFF' },
  footer:      { textAlign: 'center', fontSize: 11, color: '#CCC' },
});